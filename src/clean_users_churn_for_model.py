
# код достает из БД таблицу clean_users_charn (params.yaml) и сохраняет clean_users_churn_for_model.csv

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import yaml

def create_connection():
    load_dotenv()
    host = os.getenv("DB_DESTINATION_HOST")
    port = os.getenv("DB_DESTINATION_PORT")
    db_name = os.getenv("DB_DESTINATION_NAME")
    username = os.getenv("DB_DESTINATION_USER")
    password = os.getenv("DB_DESTINATION_PASSWORD")
    if not all([host, port, db_name, username, password]):
        raise ValueError("Не все переменные окружения для подключения к БД заданы")
    dsn = (
        f"postgresql+psycopg2://{username}:{password}"
        f"@{host}:{port}/{db_name}?sslmode=require"
    )
    return create_engine(dsn)


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    table_name = params["db"]["table_name"]
    engine = create_connection()
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql(query, engine)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    data.to_csv("data/processed/clean_users_churn_for_model.csv", index=False)
if __name__ == "__main__":
    main()