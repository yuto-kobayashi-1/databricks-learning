# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,01_basic_evaluation
# MAGIC %md
# MAGIC # 01_basic_evaluation
# MAGIC
# MAGIC `mlflow.genai.evaluate()` の基本的な使い方を確認します。
# MAGIC
# MAGIC - **Evaluation Dataset** (Python辞書リスト)
# MAGIC - **predict_fn** (Databricks Foundation Model API)
# MAGIC - **Built-in Scorer** (Correctness, RelevanceToQuery, Guidelines)

# COMMAND ----------

# DBTITLE 1,パッケージインストール
# MAGIC %pip install openai

# COMMAND ----------

# DBTITLE 1,セットアップ
import os
import mlflow
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

# COMMAND ----------

# DBTITLE 1,Evaluation Dataset
# MAGIC %md
# MAGIC ## 1. Evaluation Dataset
# MAGIC
# MAGIC Pythonの辞書リストでもEvaluation Datasetとして利用可能です。
# MAGIC
# MAGIC `inputs`のキー(`query`)が`predict_fn`の引数名に対応し、
# MAGIC `expectations`にはGround Truth（期待する事実）を格納します。
# MAGIC
# MAGIC | 名前 | 固定度 | 意味 |
# MAGIC |---|---|---|
# MAGIC | `inputs` | 固定 | アプリへの入力 |
# MAGIC | `expectations` | 固定 | Ground Truth / 期待値を入れる領域 |
# MAGIC | `expected_facts` | Built-in用の予約キー | `Correctness` が読む |
# MAGIC | `expected_response` | Built-in用の予約キー | `Correctness` が読む |
# MAGIC | `guidelines` | Built-in用の予約キー | `Guidelines` 系が読む |
# MAGIC | 独自キー | 自由 | Custom Scorerで使える |

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

# DBTITLE 1,predict_fn
# MAGIC %md
# MAGIC ## 2. predict_fn
# MAGIC
# MAGIC `inputs`のキーと引数名を対応させます。
# MAGIC 例: `inputs["query"]` → `predict_fn(query=...)`
# MAGIC
# MAGIC 実際のアプリケーションでは、評価用ロジックを`predict_fn`内に再実装するより、
# MAGIC 既存のアプリケーションをラップして使う方が使いやすいと思います。

# COMMAND ----------

# DBTITLE 1,predict_fn定義
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
            {
                "role": "user",
                "content": query,
            },
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

# DBTITLE 1,Scorer
# MAGIC %md
# MAGIC ## 3. Scorer
# MAGIC
# MAGIC MLflowが提供するBuilt-in Scorerを利用します。
# MAGIC 利用可能なBuilt-in Scorerの一覧は[こちら](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/?utm_source=chatgpt.com)
# MAGIC | Scorer | 役割 |
# MAGIC |---|---|
# MAGIC | `Correctness` | `expected_facts`が回答によって満たされているかをLLM Judgeで評価 |
# MAGIC | `RelevanceToQuery` | 回答が質問に対して関連しているかを評価 |
# MAGIC | `Guidelines` | 任意の評価条件を追加（今回は「日本語で回答しているか」） |

# COMMAND ----------

# DBTITLE 1,scorers定義
from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    Guidelines,
)

scorers = [
    Correctness(
        model=JUDGE_MODEL,
    ),
    RelevanceToQuery(
        model=JUDGE_MODEL,
    ),
    Guidelines(
        name="japanese_response",
        guidelines="回答が日本語で記述されていること。",
        model=JUDGE_MODEL,
    ),
]

# COMMAND ----------

# DBTITLE 1,評価の実行
# MAGIC %md
# MAGIC ## 4. 評価の実行
# MAGIC
# MAGIC Evaluationを実行し、結果を確認します。
# MAGIC
# MAGIC `value`には判定結果、`rationale`にはJudgeがその判定をした理由が記録されます。
# MAGIC
# MAGIC ```
# MAGIC correctness/value, correctness/rationale
# MAGIC relevance_to_query/value, relevance_to_query/rationale
# MAGIC japanese_response/value, japanese_response/rationale
# MAGIC ```
# MAGIC
# MAGIC MLflow UIから各レコードを開くと、Scorerの結果だけでなく、
# MAGIC その際に生成されたTraceも確認できます。
# MAGIC
# MAGIC ⚠️Built-in Scorerのプロンプトが英語表記のため、Judge

# COMMAND ----------

# DBTITLE 1,evaluate実行
result = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=scorers,
)

# 集計値
print(result.metrics)

# COMMAND ----------

# DBTITLE 1,結果確認
# 各レコードの結果
display(result.result_df)

# COMMAND ----------

