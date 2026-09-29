# DatabricksとMLflow 3でLLM/RAG評価パイプラインを実装する

本リポジトリは、MLflow 3の `mlflow.genai.evaluate()` を利用してLLM/RAGアプリケーションを評価するコード例を提供します。

## 構成

```
mlflow/genai-evaluation/
├── README.md                        # 本ファイル
├── 01_basic_evaluation.py           # mlflow.genai.evaluate()の基本使い方
├── 02_evaluation_dataset.py          # Evaluation DatasetをUnity Catalogで管理
├── 03_rag_trace.py                   # RAGのRetrieval処理をTraceとして記録
├── 04_retrieval_metrics.py           # Recall/Precision/Exact MatchのCustom Scorer
└── notebooks/
    └── genai_evaluation_demo         # 全セクションを統合したDatabricks Notebook
```

## 前提環境

- Databricks Workspace (ServerlessまたはClassic Compute)
- MLflow 3.x (Databricks Runtime 16.0+ にバンドル)
- Unity Catalogが有効な環境
- Foundation Model APIへのアクセス権限

## 実行方法

### 個別スクリプトとして実行

各 `.py` ファイルはDatabricks NotebookまたはPythonスクリプトとして実行できます。

### 統合デモとして実行

`notebooks/genai_evaluation_demo` をDatabricks Notebookとして上から順に実行すると、
基本評価からRAG評価まで全体を通して確認できます。

## 各ファイルの概要

| ファイル | 内容 |
|---|---|
| `01_basic_evaluation.py` | Evaluation Dataset、predict_fn、Built-in Scorer (Correctness, RelevanceToQuery, Guidelines) を使った基本評価 |
| `02_evaluation_dataset.py` | `mlflow.genai.datasets` でUnity Catalog上にEvaluation Datasetを作成・管理 |
| `03_rag_trace.py` | `@mlflow.trace` でRetrieval処理をRETRIEVER Spanとして記録 |
| `04_retrieval_metrics.py` | Custom ScorerでRetrieval Recall/Precision/Exact Matchを計算 |
| `notebooks/genai_evaluation_demo` | 上記を最初から最後まで動かせるDatabricks Notebook |

## 参考

- [MLflow GenAI Evaluation API](https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html)
- [MLflow GenAI Evaluation Quickstart](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/)
- [Databricks: Build Evaluation Dataset](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/build-eval-dataset)
- [MLflow Custom Scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/)
- [MLflow RAG Judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/)
- [MLflow Span Concepts](https://www.mlflow.org/docs/latest/genai/concepts/span/)
