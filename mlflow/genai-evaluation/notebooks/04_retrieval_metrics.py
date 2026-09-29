# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "openai",
# ]
# ///
# DBTITLE 1,04_retrieval_metrics
# MAGIC %md
# MAGIC # 04_retrieval_metrics
# MAGIC
# MAGIC 正解Documentがある場合、Retrieval Recall/Precisionや完全一致を
# MAGIC Custom Scorerで決定論的に評価します。
# MAGIC
# MAGIC - `retrieved_document_recall`
# MAGIC - `retrieved_document_precision`
# MAGIC - `exact_match`
# MAGIC
# MAGIC LLM Judgeを使う必要がなく、集合演算で一意に計算できる評価はCode-based Scorerで実装します。

# COMMAND ----------

# DBTITLE 1,パッケージインストール
# MAGIC %pip install openai

# COMMAND ----------

# DBTITLE 1,Recallの概要
# MAGIC %md
# MAGIC ## 1. Retrieved Document Recall
# MAGIC
# MAGIC 正解Documentをどれだけ取りこぼさず取得できたかを計測します。
# MAGIC
# MAGIC ```
# MAGIC 正解: doc_A, doc_B, doc_C (3件)
# MAGIC 取得: doc_A, doc_B, doc_D
# MAGIC
# MAGIC Recall = 2 / 3
# MAGIC ```
# MAGIC
# MAGIC 基本的な流れ:
# MAGIC 1. `trace.search_spans()` から `RETRIEVER` Spanを取得
# MAGIC 2. Retrieverが取得した `doc_uri` を集める
# MAGIC 3. `expectations["relevant_document_urls"]` と比較
# MAGIC 4. Recallを `Feedback` として返す

# COMMAND ----------

# DBTITLE 1,retrieved_document_recall
from mlflow.entities import Feedback, SpanType, Trace
from mlflow.genai import scorer


@scorer
def retrieved_document_recall(
    trace: Trace,
    expectations: dict,
) -> Feedback:

    retriever_spans = trace.search_spans(
        span_type=SpanType.RETRIEVER
    )

    if not retriever_spans:
        return Feedback(
            value=0,
            rationale="Retriever Spanが存在しません。",
        )

    all_document_urls = []

    for span in retriever_spans:
        all_document_urls.extend(
            [
                document["metadata"]["doc_uri"]
                for document in span.outputs
            ]
        )

    expected_document_urls = expectations[
        "relevant_document_urls"
    ]

    true_positives = len(
        set(all_document_urls)
        & set(expected_document_urls)
    )

    expected_positives = len(
        expected_document_urls
    )

    recall = (
        true_positives
        / expected_positives
    )

    return Feedback(
        value=recall,
        rationale=(
            f"正解Document {expected_positives}件のうち"
            f"{true_positives}件を取得しました。"
        ),
    )

# COMMAND ----------

# DBTITLE 1,Precisionの概要
# MAGIC %md
# MAGIC ## 2. Retrieved Document Precision
# MAGIC
# MAGIC 取得したDocumentの中にどれだけ不要なDocumentが混ざっていないかを計測します。
# MAGIC
# MAGIC ```
# MAGIC 正解: doc_A, doc_B, doc_C
# MAGIC 取得: doc_A, doc_B, doc_D (3件)
# MAGIC
# MAGIC Precision = 2 / 3
# MAGIC ```
# MAGIC
# MAGIC Recallは「取りこぼし」を、Precisionは「ノイズ」を見る指標です。
# MAGIC RAGでは両方を見ることでRetrieverの挙動をより詳細に確認できます。

# COMMAND ----------

# DBTITLE 1,retrieved_document_precision
@scorer
def retrieved_document_precision(
    trace: Trace,
    expectations: dict,
) -> Feedback:

    retriever_spans = trace.search_spans(
        span_type=SpanType.RETRIEVER
    )

    if not retriever_spans:
        return Feedback(
            value=0,
            rationale="Retriever Spanが存在しません。",
        )

    retrieved = {
        document["metadata"]["doc_uri"]
        for span in retriever_spans
        for document in span.outputs
    }

    expected = set(
        expectations["relevant_document_urls"]
    )

    if not retrieved:
        return Feedback(
            value=0,
            rationale="Documentを取得できませんでした。",
        )

    true_positives = len(
        retrieved & expected
    )

    precision = (
        true_positives
        / len(retrieved)
    )

    return Feedback(
        value=precision,
        rationale=(
            f"取得したDocument {len(retrieved)}件のうち"
            f"{true_positives}件が正解Documentでした。"
        ),
    )

# COMMAND ----------

# DBTITLE 1,Exact Matchの概要
# MAGIC %md
# MAGIC ## 3. 回答の完全一致 (Exact Match)
# MAGIC
# MAGIC 同じ考え方は生成回答にも使えます。
# MAGIC Ground Truthと完全一致するかだけを確認したい場合、LLM Judgeは不要です。
# MAGIC
# MAGIC `@scorer` では `outputs`, `expectations`, `inputs`, `trace` などを引数として受け取れます。
# MAGIC 単純なCustom Scorerであれば `bool`, `int`, `float` などの値を直接返すことも可能です。

# COMMAND ----------

# DBTITLE 1,exact_match
@scorer
def exact_match(
    outputs: str,
    expectations: dict,
) -> bool:

    return (
        outputs
        == expectations["expected_response"]
    )

# COMMAND ----------

# DBTITLE 1,評価の概要
# MAGIC %md
# MAGIC ## 4. Custom Scorerを使った評価の実行
# MAGIC
# MAGIC Custom ScorerはBuilt-in Scorerと同じように `scorers` リストに追加して利用できます。
# MAGIC
# MAGIC `trace` を引数に取るCustom Scorerでは実行Traceが必要です。
# MAGIC 静的なPandas DataFrameだけを渡してRetriever Spanを評価することはできないため注意が必要です。

# COMMAND ----------

# DBTITLE 1,evaluate実行
import os
import mlflow
from mlflow.entities import Document, SpanType
from openai import OpenAI
from databricks.sdk import WorkspaceClient
from mlflow.genai.scorers import Correctness, RelevanceToQuery

JUDGE_MODEL = "databricks:/databricks-gpt-oss-120b"
MODEL_NAME = "databricks-gpt-oss-120b"

w = WorkspaceClient()
auth_headers = w.config.authenticate()
token = auth_headers["Authorization"].replace("Bearer ", "")
client = OpenAI(
    api_key=token,
    base_url=f"{w.config.host}/serving-endpoints",
)

# --- RAGアプリケーション（RETRIEVER Span付き） ---

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_documents(query: str):
    # ダミーデータ（実環境ではVector Searchの結果に置き換える）
    search_results = [
        {"text": "Delta Lakeはオープンソースのストレージレイヤーである。", "document_uri": "doc_A", "chunk_id": "chunk_1"},
        {"text": "Delta LakeはACIDトランザクションをサポートする。", "document_uri": "doc_A", "chunk_id": "chunk_2"},
        {"text": "Unity Catalogはデータガバナンスを提供する。", "document_uri": "doc_B", "chunk_id": "chunk_3"},
    ]

    trace_outputs = [
        Document(
            page_content=result["text"],
            metadata={"doc_uri": result["document_uri"], "chunk_id": result["chunk_id"]},
        )
        for result in search_results
    ]

    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_outputs(trace_outputs)

    return trace_outputs


@mlflow.trace
def rag_app(query: str) -> str:
    documents = retrieve_documents(query)
    context = "\n\n".join([doc.page_content for doc in documents])
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": f"以下のコンテキストに基づいて回答してください。\n\n{context}"},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content

# --- 評価の実行 ---

eval_data = [
    {
        "inputs": {"query": "Delta Lakeとは何ですか？"},
        "expectations": {
            "expected_facts": [
                "Delta Lakeはオープンソースのストレージレイヤーである",
                "ACIDトランザクションをサポートする",
            ],
            "relevant_document_urls": ["doc_A", "doc_B", "doc_C"],
        },
    },
]

def predict_fn(query: str) -> str:
    return rag_app(query=query)

result = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[
        Correctness(model=JUDGE_MODEL),
        RelevanceToQuery(model=JUDGE_MODEL),
        retrieved_document_recall,
        retrieved_document_precision,
    ],
)

print(result.metrics)
display(result.result_df[["trace_id", "correctness/value", "relevance_to_query/value", "retrieved_document_recall/value", "retrieved_document_precision/value"]].to_string())

# COMMAND ----------

# DBTITLE 1,Trace情報をカラムに追加する
# MAGIC %md
# MAGIC    
# MAGIC ## 5. Trace情報（Retrieval結果と回答）をカラムに追加
# MAGIC
# MAGIC `result.result_df` は通常のPandas DataFrameなので、Trace情報から必要なデータを抽出して新しいカラムとして追加できます。
# MAGIC
# MAGIC 以下の2つのカラムを追加します：
# MAGIC
# MAGIC 1. **`retrieved_docs`**: RETRIEVER Spanから取得したDocument URIのリスト
# MAGIC 2. **`response_text`**: LLMの回答テキスト（reasoning blockを除いた純粋な回答）
# MAGIC
# MAGIC これにより、評価結果と実際のデータ（取得したDocument、回答内容）を並べて表示でき、
# MAGIC 評価結果の解釈やデバッグが容易になります。

# COMMAND ----------

# DBTITLE 1,Retrieval結果と回答をカラムに追加
import json

# 取得したDocument URIのリストを抽出
def extract_retrieved_docs(spans):
    """spansカラムからRETRIEVER Spanの出力を抽出"""
    for span in spans:
        attrs = span.get("attributes", {})
        # span_typeは '"RETRIEVER"' のようにダブルクォートを含む
        if attrs.get("mlflow.spanType") == '"RETRIEVER"':
            outputs_str = attrs.get("mlflow.spanOutputs", "[]")
            # outputsはJSON文字列なのでパースが必要
            if isinstance(outputs_str, str):
                try:
                    outputs = json.loads(outputs_str)
                except:
                    return []
            else:
                outputs = outputs_str
            
            return [doc["metadata"]["doc_uri"] for doc in outputs if isinstance(doc, dict) and "metadata" in doc]
    return []

# 回答テキストを抽出（content blockからtype="text"のみ）
def extract_response_text(response):
    """responseカラムからテキストのみ抽出"""
    if isinstance(response, list):
        text_parts = []
        for part in response:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "".join(text_parts)
    return str(response) if response else ""

# カラムを追加
result.result_df["retrieved_docs"] = result.result_df["spans"].apply(extract_retrieved_docs)
result.result_df["response_text"] = result.result_df["response"].apply(extract_response_text)

# 追加したカラムを含めて表示
display(
    result.result_df[[
        "trace_id",
        "retrieved_docs",
        "response_text",
        "correctness/value",
        "relevance_to_query/value",
        "retrieved_document_recall/value",
        "retrieved_document_precision/value",
    ]]
)

# COMMAND ----------

# DBTITLE 1,使い分けまとめ
# MAGIC %md
# MAGIC ## 6. LLM JudgeとコードベースScorerを使い分ける
# MAGIC
# MAGIC | 評価したい内容 | 評価方法 |
# MAGIC |---|---|
# MAGIC | 期待する事実を回答が含んでいるか | `Correctness` (LLM Judge) |
# MAGIC | 質問に適切に回答しているか | `RelevanceToQuery` (LLM Judge) |
# MAGIC | 独自の文章・業務ルールを満たすか | `Guidelines` (LLM Judge) |
# MAGIC | Retrieved DocumentがQueryに関連するか | `RetrievalRelevance` (LLM Judge) |
# MAGIC | 回答がRetrieved DocumentにGroundingされているか | `RetrievalGroundedness` (LLM Judge) |
# MAGIC | Retrieved Documentだけで必要な情報が揃っているか | `RetrievalSufficiency` (LLM Judge) |
# MAGIC | 正解Documentをどれだけ取得できたか | Custom Scorer / Recall (Code-based) |
# MAGIC | 取得Documentに不要なものがどれだけ少ないか | Custom Scorer / Precision (Code-based) |
# MAGIC | Ground Truthとの完全一致 | Custom Scorer / Exact Match (Code-based) |
# MAGIC
# MAGIC **意味的な品質はLLM Judgeで評価し、正解が明確に定義できるものはコードで評価する。**
# MAGIC この使い分けが、LLM/RAGアプリケーションの評価パイプラインを実運用に載せる上で重要です。

# COMMAND ----------

