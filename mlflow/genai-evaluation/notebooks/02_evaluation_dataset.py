# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,02_evaluation_dataset
# MAGIC %md
# MAGIC # 02_evaluation_dataset
# MAGIC
# MAGIC Evaluation DatasetをUnity Catalogで管理します。
# MAGIC
# MAGIC - `mlflow.genai.datasets.create_dataset()`
# MAGIC - `dataset.merge_records()`
# MAGIC - `mlflow.genai.datasets.get_dataset()`
# MAGIC
# MAGIC UC上で管理することで、バージョニング・リネージ・共有・ガバナンスが利用可能になります。
# MAGIC また、一度作成した後にもデータセットへのレコード追加も可能です。

# COMMAND ----------

# DBTITLE 1,パッケージインストール
# MAGIC %pip install openai

# COMMAND ----------

# DBTITLE 1,Dataset作成
# MAGIC %md
# MAGIC ## 1. Evaluation Datasetの作成
# MAGIC
# MAGIC Unity Catalog上にEvaluation Datasetを作成します。
# MAGIC カタログ名・スキーマ名は環境に合わせて変更してください。

# COMMAND ----------

# DBTITLE 1,create_dataset
import mlflow
from mlflow.genai.datasets import create_dataset, get_dataset

# カタログ名・スキーマ名は環境に合わせて変更してください
CATALOG = "workspace"
SCHEMA = "default"
DATASET_NAME = f"{CATALOG}.{SCHEMA}.rag_eval_dataset"

try:
    dataset = create_dataset(
        name=DATASET_NAME,
    )
    print(f"Dataset created: {dataset.name}")
except Exception as e:
    if "ALREADY_EXISTS" in str(e):
        dataset = get_dataset(name=DATASET_NAME)
        print(f"Dataset already exists, retrieved: {dataset.name}")
    else:
        raise

# COMMAND ----------

# DBTITLE 1,レコード追加
# MAGIC %md
# MAGIC ## 2. 評価データの追加
# MAGIC
# MAGIC Python辞書リストと同じ形式でレコードを追加します。

# COMMAND ----------

# DBTITLE 1,merge_records
records = [
    {
        "inputs": {
            "query": "Delta Lakeとは何ですか？"
        },
        "expectations": {
            "expected_facts": [
                "Delta Lakeはオープンソースのストレージレイヤーである",
                "ACIDトランザクションをサポートする",
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

dataset = dataset.merge_records(records)

print(f"Records merged. Dataset: {dataset.name}")

# COMMAND ----------

# DBTITLE 1,Dataset取得
# MAGIC %md
# MAGIC ## 3. 既存Datasetの取得
# MAGIC
# MAGIC 別セッションや別ノートブックから利用する場合は `get_dataset()` で取得します。

# COMMAND ----------

# DBTITLE 1,get_dataset
dataset = get_dataset(
    name=DATASET_NAME,
)

print(f"Dataset retrieved: {dataset.name}")

# COMMAND ----------

# DBTITLE 1,評価の概要
# MAGIC %md
# MAGIC ## 4. Evaluation Datasetを使った評価
# MAGIC
# MAGIC MLflow Evaluation Datasetは `mlflow.genai.evaluate()` の `data` 引数に直接渡せます。

# COMMAND ----------

# DBTITLE 1,predict_fn定義
import os
from openai import OpenAI
from databricks.sdk import WorkspaceClient
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines

JUDGE_MODEL = "databricks:/databricks-gpt-oss-120b"

w = WorkspaceClient()
auth_headers = w.config.authenticate()
token = auth_headers["Authorization"].replace("Bearer ", "")
client = OpenAI(
    api_key=token,
    base_url=f"{w.config.host}/serving-endpoints",
)

def predict_fn(query: str) -> str:
    response = client.chat.completions.create(
        model="databricks-gpt-oss-120b",
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
    content = response.choices[0].message.content
    # gpt-oss-120bなどはcontent blockのリストを返す（reasoning + text）
    # type == "text" のみ抽出して文字列化する
    if isinstance(content, list):
        parts = []
        for part in content:
            # dictの場合
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
            # Pydantic modelの場合
            elif getattr(part, "type", None) == "text":
                parts.append(getattr(part, "text", ""))
        content = "".join(parts)
    return content

# COMMAND ----------

# DBTITLE 1,evaluate実行
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
    data=dataset,
    predict_fn=predict_fn,
    scorers=scorers,
)

print(result.metrics)
display(result.result_df)

# COMMAND ----------

# DBTITLE 1,継続運用
# MAGIC %md
# MAGIC ## 5. 継続的な運用イメージ
# MAGIC
# MAGIC 本番で発見された失敗ケースをEvaluation Datasetへ追加していくことで、
# MAGIC Golden Datasetとして蓄積し、Regression Evaluationに活用できます。
# MAGIC
# MAGIC ```python
# MAGIC new_failure_cases = [
# MAGIC     {
# MAGIC         "inputs": {"query": "新しい失敗ケース"},
# MAGIC         "expectations": {"expected_facts": ["..."]},
# MAGIC     }
# MAGIC ]
# MAGIC dataset = dataset.merge_records(new_failure_cases)
# MAGIC ```

# COMMAND ----------

