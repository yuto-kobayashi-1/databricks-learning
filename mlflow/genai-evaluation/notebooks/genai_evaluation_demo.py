# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # DatabricksとMLflow 3でLLM/RAG評価パイプラインを実装する
# MAGIC
# MAGIC GenAI Evaluation Demo Notebook
# MAGIC
# MAGIC 本ノートブックは以下の全セクションを最初から最後まで動かせるように統合したものです。
# MAGIC
# MAGIC 1. **基本評価** - `mlflow.genai.evaluate()` の基本
# MAGIC 2. **Evaluation Dataset** - Unity Catalogでの管理
# MAGIC 3. **RAG Trace** - Retrieval処理をTraceとして記録
# MAGIC 4. **Retrieval Metrics** - Custom ScorerでRecall/Precision/Exact Match
# MAGIC
# MAGIC 参考:
# MAGIC - https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html
# MAGIC - https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/build-eval-dataset

# COMMAND ----------

# DBTITLE 1,パッケージインストール
# MAGIC %pip install openai

# COMMAND ----------

# DBTITLE 1,セットアップ
# =============================================================
# セットアップ
# =============================================================
import os
import mlflow
from mlflow.entities import Document, Feedback, SpanType, Trace
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines
from openai import OpenAI
from databricks.sdk import WorkspaceClient

# Databricks SDKで認証トークンを取得し、OpenAIクライアントでFoundation Model APIを呼び出す
w = WorkspaceClient()
auth_headers = w.config.authenticate()
token = auth_headers["Authorization"].replace("Bearer ", "")
client = OpenAI(
    api_key=token,
    base_url=f"{w.config.host}/serving-endpoints",
)

# 評価対象モデルとLLM-as-a-Judgeモデル
MODEL_NAME = "databricks-gpt-oss-120b"
JUDGE_MODEL = "databricks:/databricks-gpt-oss-120b"

print(f"MLflow version: {mlflow.__version__}")
print(f"Model: {MODEL_NAME}")
print(f"Judge: {JUDGE_MODEL}")

# COMMAND ----------

# DBTITLE 1,セクション1: 基本評価
# MAGIC %md
# MAGIC ---
# MAGIC ## 1. `mlflow.genai.evaluate()` を使った基本評価
# MAGIC
# MAGIC Evaluation Dataset、predict_fn、Built-in Scorerを組み合わせてLLMアプリケーションを評価します。

# COMMAND ----------

# DBTITLE 1,1.1 Evaluation Dataset
# MAGIC %md
# MAGIC ### 1.1 Evaluation Dataset
# MAGIC
# MAGIC Pythonの辞書リストでもEvaluation Datasetとして利用可能です。
# MAGIC `inputs`のキーが`predict_fn`の引数名に対応し、`expectations`にはGround Truthを格納します。

# COMMAND ----------

# DBTITLE 1,eval_data定義
eval_data = [
    {
        "inputs": {
            "query": "Delta Lakeとは何ですか？"
        },
        "expectations": {
            "expected_facts": [
                "Delta Lakeはオープンソースのストレージレイヤーである",
                "ACIDトランザクションをサポートする",
                "Parquetファイルフォーマットをベースとする",
            ]
        },
    },
    {
        "inputs": {
            "query": "Unity Catalogの主な機能は何ですか？"
        },
        "expectations": {
            "expected_facts": [
                "Unity Catalogはデータガバナンスを提供する",
                "データやAI資産のアクセス制御を一元管理する",
            ]
        },
    },
]

# COMMAND ----------

# DBTITLE 1,1.2 predict_fn
# MAGIC %md
# MAGIC ### 1.2 predict_fn
# MAGIC
# MAGIC `inputs`のキーと引数名を対応させます。実際のアプリケーションでは薄いAdapterにするのが推奨です。

# COMMAND ----------

# DBTITLE 1,predict_fn定義
def extract_text(content) -> str:
    """content blockからtype=textの内容だけを抽出する。"""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
            elif getattr(part, "type", None) == "text":
                parts.append(getattr(part, "text", ""))
        return "".join(parts)

    return str(content) if content is not None else ""


def predict_fn(query: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは丁寧な日本語で回答するアシスタントです。"
                    "Databricksに関する質問に正確に答えてください。"
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    return extract_text(response.choices[0].message.content)

# COMMAND ----------

# DBTITLE 1,1.3 Scorer
# MAGIC %md
# MAGIC ### 1.3 Scorerと評価の実行
# MAGIC
# MAGIC Built-in Scorer（Correctness, RelevanceToQuery, Guidelines）を利用します。

# COMMAND ----------

# DBTITLE 1,基本評価の実行
scorers = [
    Correctness(model=JUDGE_MODEL),
    RelevanceToQuery(model=JUDGE_MODEL),
    Guidelines(
        name="japanese_response",
        guidelines="回答が日本語で記述されていること。",
        model=JUDGE_MODEL,
    ),
]

result = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=scorers,
)

print(result.metrics)
display(result.result_df)

# COMMAND ----------

# DBTITLE 1,セクション2: UC Evaluation Dataset
# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Evaluation DatasetをUnity Catalogで管理する
# MAGIC
# MAGIC 継続的に評価する場合、「どの評価データを利用したのか」「どのケースを追加したのか」
# MAGIC 「アプリケーションの変更で品質が劣化していないか」といった管理が必要です。
# MAGIC
# MAGIC DatabricksではMLflow Evaluation DatasetをUnity Catalog上で管理できます。

# COMMAND ----------

# DBTITLE 1,Dataset作成とレコード追加
from mlflow.genai.datasets import create_dataset, get_dataset

# カタログ名・スキーマ名は環境に合わせて変更してください
CATALOG = "workspace"
SCHEMA = "default"
DATASET_NAME = f"{CATALOG}.{SCHEMA}.rag_eval_dataset"

# Dataset作成
dataset = create_dataset(name=DATASET_NAME)
print(f"Dataset created: {dataset.name}")

# レコード追加
dataset = dataset.merge_records(eval_data)
print(f"Records merged. Dataset: {dataset.name}")

# COMMAND ----------

# DBTITLE 1,UC Datasetで評価
# 既存Datasetの取得（別セッションからの利用を想定）
dataset = get_dataset(name=DATASET_NAME)
print(f"Dataset retrieved: {dataset.name}")

# UC Datasetを使った評価
result = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=scorers,
)

print(result.metrics)
display(result.result_df)

# COMMAND ----------

# DBTITLE 1,セクション3: RAG Trace
# MAGIC %md
# MAGIC ---
# MAGIC ## 3. RAGのRetrievalをTraceとして記録する
# MAGIC
# MAGIC RAGでは回答が誤っていた場合、原因がGenerationとは限りません。
# MAGIC Retrieval処理を `RETRIEVER` SpanとしてTraceすることで、
# MAGIC MLflowのRAG向けBuilt-in JudgeやCustom Scorerから参照できます。

# COMMAND ----------

# DBTITLE 1,RAGアプリ定義
@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_documents(query: str):
    # 実際の環境ではVector Searchエンドポイントを呼び出す
    # ダミーデータ（実環境ではVector Searchの結果に置き換える）
    search_results = [
        {
            "text": "Delta Lakeはオープンソースのストレージレイヤーである。",
            "document_uri": "doc_A",
            "chunk_id": "chunk_1",
        },
        {
            "text": "Delta LakeはACIDトランザクションをサポートする。",
            "document_uri": "doc_B",
            "chunk_id": "chunk_2",
        },
        {
            "text": "Unity Catalogはデータガバナンスを提供する。",
            "document_uri": "doc_D",
            "chunk_id": "chunk_3",
        },
    ]

    span = mlflow.get_current_active_span()
    trace_outputs = [
        Document(
            page_content=result["text"],
            metadata={
                "doc_uri": result["document_uri"],
                "chunk_id": result["chunk_id"],
            },
        )
        for result in search_results
    ]
    if span is not None:
        span.set_outputs(trace_outputs)
    return trace_outputs


@mlflow.trace(span_type=SpanType.CHAT_MODEL)
def generate_answer(query: str, context: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは丁寧な日本語で回答するアシスタントです。"
                    "以下のコンテキストに基づいてDatabricksに関する質問に正確に答えてください。\n\n"
                    f"コンテキスト:\n{context}"
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    return extract_text(response.choices[0].message.content)


@mlflow.trace
def rag_app(query: str) -> str:
    search_results = retrieve_documents(query)
    context = "\n\n".join([doc.page_content for doc in search_results])
    answer = generate_answer(query, context)
    return answer

# COMMAND ----------

# DBTITLE 1,RAG動作確認
# 動作確認
answer = rag_app("Delta Lakeとは何ですか？")
print(answer)

# COMMAND ----------

# DBTITLE 1,セクション4: Retrieval Metrics
# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Retrieval Recall / Precision / Exact MatchをCustom Scorerで評価する
# MAGIC
# MAGIC Ground Truthとなる正解Documentが存在する場合、集合演算で一意に計算できます。
# MAGIC LLM Judgeを使う必要がなく、Code-based Scorerで実装します。

# COMMAND ----------

# DBTITLE 1,retrieved_document_recall
@scorer
def retrieved_document_recall(
    trace: Trace,
    expectations: dict,
) -> Feedback:

    retriever_spans = trace.search_spans(span_type=SpanType.RETRIEVER)

    if not retriever_spans:
        return Feedback(value=0, rationale="Retriever Spanが存在しません。")

    all_document_urls = []
    for span in retriever_spans:
        all_document_urls.extend(
            [document["metadata"]["doc_uri"] for document in span.outputs]
        )

    expected_document_urls = expectations["relevant_document_urls"]
    true_positives = len(set(all_document_urls) & set(expected_document_urls))
    expected_positives = len(expected_document_urls)
    recall = true_positives / expected_positives

    return Feedback(
        value=recall,
        rationale=f"正解Document {expected_positives}件のうち{true_positives}件を取得しました。",
    )

# COMMAND ----------

# DBTITLE 1,retrieved_document_precision
@scorer
def retrieved_document_precision(
    trace: Trace,
    expectations: dict,
) -> Feedback:

    retriever_spans = trace.search_spans(span_type=SpanType.RETRIEVER)

    if not retriever_spans:
        return Feedback(value=0, rationale="Retriever Spanが存在しません。")

    retrieved = {
        document["metadata"]["doc_uri"]
        for span in retriever_spans
        for document in span.outputs
    }
    expected = set(expectations["relevant_document_urls"])

    if not retrieved:
        return Feedback(value=0, rationale="Documentを取得できませんでした。")

    true_positives = len(retrieved & expected)
    precision = true_positives / len(retrieved)

    return Feedback(
        value=precision,
        rationale=f"取得したDocument {len(retrieved)}件のうち{true_positives}件が正解Documentでした。",
    )

# COMMAND ----------

# DBTITLE 1,exact_match
@scorer
def exact_match(
    outputs: str,
    expectations: dict,
) -> bool:
    return outputs == expectations["expected_response"]

# COMMAND ----------

# DBTITLE 1,RAG評価の実行
# RAGアプリケーションを評価
rag_eval_data = [
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

def rag_predict_fn(query: str) -> str:
    return rag_app(query=query)

result = mlflow.genai.evaluate(
    data=rag_eval_data,
    predict_fn=rag_predict_fn,
    scorers=[
        Correctness(model=JUDGE_MODEL),
        RelevanceToQuery(model=JUDGE_MODEL),
        retrieved_document_recall,
        retrieved_document_precision,
    ],
)

print(result.metrics)
print(result.result_df[["trace_id", "correctness/value", "relevance_to_query/value", "retrieved_document_recall/value", "retrieved_document_precision/value"]].to_string())

# COMMAND ----------

# DBTITLE 1,まとめ
# MAGIC %md
# MAGIC ---
# MAGIC ## 5. LLM JudgeとコードベースScorerを使い分ける
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

