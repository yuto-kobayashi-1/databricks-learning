# Databricks notebook source
# DBTITLE 1,03_rag_trace
# MAGIC %md
# MAGIC # 03_rag_trace
# MAGIC
# MAGIC RAGのRetrieval処理をTraceとして記録します。
# MAGIC
# MAGIC - `@mlflow.trace` デコレータ
# MAGIC - `SpanType.RETRIEVER`
# MAGIC - `Document` オブジェクト
# MAGIC - Retriever出力のTrace記録
# MAGIC
# MAGIC RAGでは回答が誤っていた場合、原因がGenerationとは限りません。
# MAGIC Retrievalを独立して評価するために、RETRIEVER Spanとして記録しておきます。

# COMMAND ----------

# DBTITLE 1,パッケージインストール
# MAGIC %pip install openai

# COMMAND ----------

# DBTITLE 1,RETRIEVER Spanの概要
# MAGIC %md
# MAGIC ## 1. Retrieval処理をRETRIEVER Spanとして記録する
# MAGIC
# MAGIC MLflowではRetrieval処理を `span_type=SpanType.RETRIEVER` としてTraceできます。
# MAGIC
# MAGIC Retriever Spanの出力には、取得したDocumentを記録します。
# MAGIC MLflowが推奨するRetriever Span Schemaでは、
# MAGIC `Document`に`page_content`を持ち、`metadata`に`doc_uri`や`chunk_id`などを格納します。

# COMMAND ----------

# DBTITLE 1,セットアップ
import os
import mlflow
from mlflow.entities import Document, SpanType

from openai import OpenAI
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
auth_headers = w.config.authenticate()
token = auth_headers["Authorization"].replace("Bearer ", "")
client = OpenAI(
    api_key=token,
    base_url=f"{w.config.host}/serving-endpoints",
)

MODEL_NAME = "databricks-gpt-oss-120b"

# COMMAND ----------

# DBTITLE 1,Retriever関数の概要
# MAGIC %md
# MAGIC ## 2. Vector Searchを呼び出すRetriever関数
# MAGIC
# MAGIC `@mlflow.trace(span_type=SpanType.RETRIEVER)` でデコレートし、
# MAGIC 取得したDocumentを `span.set_outputs()` でTraceに記録します。
# MAGIC
# MAGIC 以下の例ではダミーのVector Search呼び出しを想定しています。

# COMMAND ----------

# DBTITLE 1,retrieve_documents
@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_documents(query: str):
    # 実際の環境ではVector Searchエンドポイントを呼び出す
    # search_results = vector_search_index.similarity_search(query, num_results=5)
    
    # ダミーデータ（実環境ではVector Searchの結果に置き換える）
    search_results = [
        {
            "text": "Delta Lakeはオープンソースのストレージレイヤーであり、ACIDトランザクションを提供する。",
            "document_uri": "doc_A",
            "chunk_id": "chunk_1",
        },
        {
            "text": "Delta LakeはParquetファイルフォーマットをベースとしている。",
            "document_uri": "doc_A",
            "chunk_id": "chunk_2",
        },
        {
            "text": "Unity Catalogはデータガバナンスを提供し、アクセス制御を一元管理する。",
            "document_uri": "doc_B",
            "chunk_id": "chunk_3",
        },
    ]

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

    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_outputs(trace_outputs)

    return trace_outputs

# COMMAND ----------

# DBTITLE 1,動作確認
# 動作確認: retrieve_documentsを呼び出してTraceを確認する
results = retrieve_documents("Delta Lakeとは何ですか？")
print(f"Retrieved {len(results)} documents")
for r in results:
    print(f"  - doc_uri: {r['document_uri']}, chunk_id: {r['chunk_id']}")

# COMMAND ----------

# DBTITLE 1,RAGアプリケーションの概要
# MAGIC %md
# MAGIC ## 3. RAGアプリケーション全体のTrace
# MAGIC
# MAGIC Retriever Spanを含むRAGアプリケーションを定義し、
# MAGIC `predict_fn` として評価に組み込みます。
# MAGIC
# MAGIC ```mermaid
# MAGIC flowchart TD
# MAGIC     A["RAG Trace"]
# MAGIC     B["RETRIEVER Span"]
# MAGIC     C["CHAT_MODEL Span"]
# MAGIC     A --> B
# MAGIC     A --> C
# MAGIC     B --> D["Retrieved Documents"]
# MAGIC     C --> E["Generated Answer"]
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,rag_app定義
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
    return response.choices[0].message.content


@mlflow.trace
def rag_app(query: str) -> str:
    # Retrieval
    documents = retrieve_documents(query)
    context = "\n\n".join([doc.page_content for doc in documents])

    # Generation
    answer = generate_answer(query, context)
    return answer

# COMMAND ----------

# DBTITLE 1,RAG動作確認
# 動作確認
answer = rag_app("Delta Lakeとは何ですか？")
print(answer)

# COMMAND ----------

# DBTITLE 1,評価への組み込み
# MAGIC %md
# MAGIC ## 4. 評価への組み込み
# MAGIC
# MAGIC このRAGアプリケーションを `predict_fn` として評価に組み込みます。
# MAGIC Retriever SpanがTraceに記録されているため、
# MAGIC MLflowのRAG向けBuilt-in Judge（`RetrievalRelevance`, `RetrievalGroundedness`, `RetrievalSufficiency`）が利用できます。
# MAGIC また、Custom ScorerからTrace内のRETRIEVER Spanを参照することも可能です。

# COMMAND ----------

# DBTITLE 1,evaluate実行
from mlflow.genai.scorers import Correctness, RelevanceToQuery

JUDGE_MODEL = "databricks:/databricks-gpt-oss-120b"

eval_data = [
    {
        "inputs": {"query": "Delta Lakeとは何ですか？"},
        "expectations": {
            "expected_facts": [
                "Delta Lakeはオープンソースのストレージレイヤーである",
                "ACIDトランザクションをサポートする",
            ],
            "relevant_document_urls": ["doc_A"],
        },
    },
]

# predict_fnはrag_appを呼び出す薄いAdapter
def predict_fn(query: str) -> str:
    return rag_app(query=query)

result = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[
        Correctness(model=JUDGE_MODEL),
        RelevanceToQuery(model=JUDGE_MODEL),
    ],
)

print(result.metrics)
display(result.result_df)

# COMMAND ----------

