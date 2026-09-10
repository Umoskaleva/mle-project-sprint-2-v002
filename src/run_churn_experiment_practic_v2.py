import os
from datetime import datetime

import mlflow
import pandas as pd
import boto3
import joblib
import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, log_loss



# Каждый запуск получит новое имя эксперимента
EXPERIMENT_NAME = f"churn_uliana_v3_1"
RUN_NAME = "log_model"

# Настройка переменных окружения для Yandex Cloud S3
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"  # endpoint бакета от YandexCloud
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")  # получаем id ключа бакета из .env
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")  # получаем ключ бакета из .env

# Создаем клиент S3
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"]
)

# скачиваем модель в локальный файл
s3.download_file(
    os.getenv("AWS_BUCKET_NAME"),
    "models/churn/logreg_churn_model.joblib",
    "logreg_churn_model.joblib"
)

# загружаем в переменную model
model = joblib.load("logreg_churn_model.joblib")

