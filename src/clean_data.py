# код чистить data/processed/clean_users_churn_for_model.csv от дубликатов, пропусков и выбросов

import pandas as pd
import numpy as np
from pathlib import Path


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет полные дубликаты строк.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Удалено дубликатов: {before - after} (осталось {after} строк)")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет ненужные для модели признаки.
    """
    cols_to_drop = ['id', 'customer_id', 'begin_date', 'end_date']
    # Проверяем, какие колонки есть в данных
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    if cols_to_drop:
        print(f"Удалены колонки: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    else:
        print("Колонки для удаления не найдены")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обрабатывает пропуски:
    - для числовых колонок: заполняем медианой;
    - для категориальных (object/category): заполняем модой;
    - если вся колонка состоит из NaN — удаляем её.
    """
    # Удаляем колонки, где все значения NaN
    cols_all_nan = [col for col in df.columns if df[col].isna().all()]
    if cols_all_nan:
        print(f"Удалены колонки, полностью состоящие из NaN: {cols_all_nan}")
        df = df.drop(columns=cols_all_nan)

    # Числовые колонки
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Категориальные колонки (object, category)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                # Если моды нет (все NaN), удаляем колонку
                df = df.drop(columns=[col])

    return df


def remove_outliers_iqr(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Удаляет выбросы в числовых колонках методом IQR.
    IQR = Q3 - Q1.
    Границы: [Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR].
    """
    before = len(df)
    num_cols = df.select_dtypes(include=[np.number]).columns

    mask = pd.Series(True, index=df.index)

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR

        col_mask = (df[col] >= lower) & (df[col] <= upper)
        mask &= col_mask

    df = df[mask].reset_index(drop=True)
    after = len(df)

    print(f"Удалено выбросов (IQR): {before - after} строк (осталось {after} строк)")
    return df


def main():
    input_path = Path(
        "/home/mle-user/mle_projects/mle-project-sprint-2-v002/"
        "data/processed/clean_users_churn_for_model.csv"
    )

    output_path = Path(
        "/home/mle-user/mle_projects/mle-project-sprint-2-v002/"
        "data/processed/clean_users_churn_for_model_cleaned.csv"
    )

    print("Загрузка данных...")
    df = pd.read_csv(input_path)

    print("Исходный размер:", df.shape)
    print("Колонки:", df.columns.tolist())

    # 1. Удаление ненужных признаков
    print("\n=== Удаление ненужных признаков ===")
    df = drop_unnecessary_columns(df)
    print(f"Размер после удаления колонок: {df.shape}")

    # 2. Дубликаты
    print("\n=== Удаление дубликатов ===")
    df = remove_duplicates(df)

    # 3. Пропуски
    print("\n=== Обработка пропусков ===")
    df = handle_missing_values(df)
    print("Пропусков после обработки:", df.isna().sum().sum())

    # 4. Выбросы
    print("\n=== Удаление выбросов (IQR) ===")
    df = remove_outliers_iqr(df, iqr_multiplier=1.5)

    print("\nИтоговый размер:", df.shape)
    print("Колонки после очистки:", df.columns.tolist())

    # Сохранение
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nОчищенные данные сохранены в:\n{output_path}")


if __name__ == "__main__":
    main()