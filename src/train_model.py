import json
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

import os
import boto3
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
    make_scorer,
)


# ---------- Построение scoring-словаря для CV ----------
def build_scoring_dict(metrics_list):
    """
    metrics_list — список строк из params.yaml, например:
    ['roc_auc', 'f1', 'precision', 'recall', 'logloss']
    """
    scoring = {}
    for metric in metrics_list:
        if metric == "roc_auc":
            scoring["roc_auc"] = "roc_auc"
        elif metric == "f1":
            scoring["f1"] = "f1"
        elif metric == "precision":
            scoring["precision"] = "precision"
        elif metric == "recall":
            scoring["recall"] = "recall"
        elif metric in ("logloss", "neg_log_loss"):
            scoring["neg_log_loss"] = "neg_log_loss"
        else:
            raise ValueError(f"Неизвестная метрика для CV: {metric}")
    return scoring


# ---------- Метрики на тесте ----------
def compute_test_metrics(y_true, y_pred, y_proba, metrics_list):
    """
    metrics_list — список из params.yaml (test_metrics).
    Возвращает dict с результатами.
    """
    res = {}
    for m in metrics_list:
        if m == "roc_auc":
            res["roc_auc"] = roc_auc_score(y_true, y_proba)
        elif m == "f1":
            res["f1"] = f1_score(y_true, y_pred, average="macro")
        elif m == "precision":
            res["precision"] = precision_score(y_true, y_pred, average="macro")
        elif m == "recall":
            res["recall"] = recall_score(y_true, y_pred, average="macro")
        elif m == "logloss":
            res["logloss"] = log_loss(y_true, y_proba)
        elif m == "confusion_matrix":
            res["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        else:
            raise ValueError(f"Неизвестная метрика для теста: {m}")
    return res


# ---------- main ----------
def main():
    # 1. Чтение params.yaml
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    target_col = params["split"]["target_col"]
    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    model_params = params["model"]["hyperparams"]
    cv_params = params["cv"]

    # 2. Загрузка данных из CSV
    data = pd.read_csv("data/processed/clean_users_churn_for_model_preprocessed.csv")

    # Проверка, что целевая колонка есть
    if target_col not in data.columns:
        raise ValueError(f"Целевая колонка '{target_col}' не найдена в датасете")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 3. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # сохраняем долю классов в train и test
    )

    # 4. Модель логистической регрессии
    # n_jobs здесь не нужен (и часто игнорируется)
    model = LogisticRegression(
        max_iter=model_params["max_iter"],
        C=model_params["C"],
        random_state=random_state,
    )

    # 5. Кросс-валидация
    cv = StratifiedKFold(
        n_splits=cv_params["n_splits"],
        shuffle=True,
        random_state=random_state,
    )

    # Строим scoring-словарь из cv.metrics
    scoring = build_scoring_dict(cv_params["metrics"])

    cv_results = cross_validate(
        model,
        X,  # можно использовать весь датасет для CV
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=cv_params["n_jobs"],
        return_train_score=False,
    )

    # Средние значения по фолдам для каждой метрики
    cv_mean_scores = {}

    for metric in cv_params["metrics"]:
        key = f"test_{metric}"
        score = float(np.mean(cv_results[key]))

        if metric == "neg_log_loss":
            cv_mean_scores["logloss"] = -score
        else:
            cv_mean_scores[metric] = score

    print("\n=== CV mean scores ===")
    for metric, value in cv_mean_scores.items():
        print(f"{metric}: {value:.4f}")

    # 6. Обучение финальной модели на train
    model.fit(X_train, y_train)

    # Предсказания на test
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # вероятность класса 1

    # 7. Метрики на тесте
    test_metrics = compute_test_metrics(
        y_test, y_pred, y_proba, cv_params["test_metrics"]
    )

    print("\n=== Test metrics ===")
    for metric, value in test_metrics.items():
        if metric == "confusion_matrix":
            print(f"{metric}:\n{np.array(value)}")
        else:
            print(f"{metric}: {value:.4f}")

    # 8. Сохранение результатов
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # CV метрики
    with open(results_dir / "cv_mean_scores.json", "w", encoding="utf-8") as f:
        json.dump(cv_mean_scores, f, indent=2)

    # Тестовые метрики
    with open(results_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Предсказания на тесте
    results_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    )
    results_df.to_csv(results_dir / "test_predictions.csv", index=False)

    # Сохранение модели (опционально)
    import joblib
    joblib.dump(model, results_dir / "logreg_churn_model.joblib")

    print("\nРезультаты сохранены в папку results/")
    print("Модель сохранена в results/logreg_churn_model.joblib")


if __name__ == "__main__":
    main()
    
    
    
    # загрузка модели в хранилище S3

# --- Настройки ---
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")  # имя бакета
MODEL_LOCAL_PATH = "/home/mle-user/mle_projects/mle-project-sprint-2-v002/results/logreg_churn_model.joblib"
S3_KEY = "models/churn/logreg_churn_model.joblib"  # Путь внутри бакета

# --- Переменные окружения для Yandex Cloud S3 ---
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_DEFAULT_REGION"] = "ru-central1"

# --- Создание клиента S3 ---
s3_client = boto3.client(
    service_name='s3',
    endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ["AWS_DEFAULT_REGION"],
)

# --- Загрузка файла ---
model_path = Path(MODEL_LOCAL_PATH)
if not model_path.exists():
    raise FileNotFoundError(f"Файл не найден: {model_path}")

s3_client.upload_file(
    Filename=str(model_path),
    Bucket=BUCKET_NAME,
    Key=S3_KEY
)

print(f"✅ Модель загружена: s3://{BUCKET_NAME}/{S3_KEY}")