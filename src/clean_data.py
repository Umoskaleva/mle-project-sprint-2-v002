# Код очищает data/processed/clean_users_churn_for_model.csv
# от ненужных колонок, дубликатов, пропусков и выбросов,
# затем подготавливает признаки для логистической регрессии.

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет полные дубликаты строк."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"Удалено дубликатов: {before - after} (осталось {after} строк)")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет технические и неинформативные для модели признаки."""
    cols_to_drop = ["id", "customer_id", "begin_date", "end_date"]

    cols_to_drop = [col for col in cols_to_drop if col in df.columns]

    if cols_to_drop:
        print(f"Удалены колонки: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    else:
        print("Колонки для удаления не найдены")

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит total_charges к числовому виду.
    Некорректные значения, например пустые строки, станут NaN.
    """
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(
            df["total_charges"],
            errors="coerce",
        )
        print("Колонка total_charges приведена к числовому типу")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обрабатывает пропуски:
    - числовые колонки: медиана;
    - категориальные колонки: наиболее частая категория;
    - колонки, полностью состоящие из NaN: удаление.

    target не заполняем: строку без известного target удаляем.
    """
    cols_all_nan = [col for col in df.columns if df[col].isna().all()]

    if cols_all_nan:
        print(f"Удалены колонки, полностью состоящие из NaN: {cols_all_nan}")
        df = df.drop(columns=cols_all_nan)

    # Целевая переменная обязательна для обучения
    if "target" in df.columns:
        before = len(df)
        df = df.dropna(subset=["target"])
        print(f"Удалено строк без target: {before - len(df)}")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # target не является признаком, поэтому не заполняем его медианой
    if "target" in num_cols:
        num_cols.remove("target")

    for col in num_cols:
        if df[col].isna().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"Пропуски в '{col}' заполнены медианой: {median_value}")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        if df[col].isna().any():
            mode_values = df[col].mode()

            if len(mode_values) > 0:
                df[col] = df[col].fillna(mode_values.iloc[0])
                print(f"Пропуски в '{col}' заполнены модой: {mode_values.iloc[0]}")
            else:
                df = df.drop(columns=[col])
                print(f"Колонка '{col}' удалена: невозможно заполнить пропуски")

    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Удаляет выбросы методом IQR только из непрерывных числовых признаков.

    target и бинарный senior_citizen исключены:
    для бинарных признаков искать выбросы нельзя.
    """
    before = len(df)

    excluded_cols = ["target", "senior_citizen"]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [col for col in num_cols if col not in excluded_cols]

    mask = pd.Series(True, index=df.index)

    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        # Если все значения одинаковы, выбросы по IQR не ищем
        if iqr == 0:
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        col_mask = df[col].between(lower, upper)
        mask &= col_mask

    df = df[mask].reset_index(drop=True)

    print(f"Удалено выбросов (IQR): {before - len(df)} (осталось {len(df)} строк)")
    return df


def preprocess_for_logistic_regression(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Готовит данные для логистической регрессии:
    - target оставляет без изменений;
    - числовые признаки масштабирует;
    - категориальные превращает в бинарные колонки через One-Hot Encoding.
    """
    target = df["target"].copy()
    X = df.drop(columns=["target"]).copy()

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\n=== Предобработка для логистической регрессии ===")
    print("Числовые признаки:", numeric_features)
    print("Категориальные признаки:", categorical_features)

    # Масштабирование числовых признаков
    if numeric_features:
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Преобразование строковых категорий в 0/1
    if categorical_features:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )

        encoded = encoder.fit_transform(X[categorical_features])

        encoded_columns = encoder.get_feature_names_out(categorical_features)

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoded_columns,
            index=X.index,
        )

        # Удаляем исходные строковые колонки и добавляем новые 0/1 колонки
        X = X.drop(columns=categorical_features)
        X = pd.concat([X, encoded_df], axis=1)

    # target возвращаем в итоговый DataFrame
    result = pd.concat(
        [
            X.reset_index(drop=True),
            target.reset_index(drop=True),
        ],
        axis=1,
    )

    print("Размер после кодирования:", result.shape)
    return result


def main():
    input_path = Path(
        "/home/mle-user/mle_projects/mle-project-sprint-2-v002/"
        "data/processed/clean_users_churn_for_model.csv"
    )

    cleaned_path = Path(
        "/home/mle-user/mle_projects/mle-project-sprint-2-v002/"
        "data/processed/clean_users_churn_for_model_cleaned.csv"
    )

    preprocessed_path = Path(
        "/home/mle-user/mle_projects/mle-project-sprint-2-v002/"
        "data/processed/clean_users_churn_for_model_preprocessed.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Исходный файл не найден:\n{input_path}")

    print("Загрузка данных...")
    df = pd.read_csv(input_path)

    print("Исходный размер:", df.shape)
    print("Исходные колонки:", df.columns.tolist())

    print("\n=== Приведение типов ===")
    df = convert_numeric_columns(df)

    print("\n=== Удаление ненужных признаков ===")
    df = drop_unnecessary_columns(df)

    print("\n=== Удаление дубликатов ===")
    df = remove_duplicates(df)

    print("\n=== Обработка пропусков ===")
    df = handle_missing_values(df)
    print("Всего пропусков после обработки:", df.isna().sum().sum())

    print("\n=== Удаление выбросов ===")
    df = remove_outliers_iqr(df, iqr_multiplier=1.5)

    print("\nРазмер очищенных данных:", df.shape)

    # Сохраняем очищенный, но ещё не закодированный набор
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_path, index=False)

    print(f"\nОчищенные данные сохранены в:\n{cleaned_path}")

    # Кодирование категорий и масштабирование чисел
    df_preprocessed = preprocess_for_logistic_regression(df)

    # Сохраняем данные, готовые для модели
    df_preprocessed.to_csv(preprocessed_path, index=False)

    print(f"\nДанные для логистической регрессии сохранены в:\n{preprocessed_path}")
    print("Итоговый размер:", df_preprocessed.shape)


if __name__ == "__main__":
    main()