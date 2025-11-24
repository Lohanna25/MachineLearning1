from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from .config import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL


def build_preprocessor(
    categorical_cols: List[str] | None = None,
    numeric_cols: List[str] | None = None,
) -> ColumnTransformer:
    """
    Crea el ColumnTransformer que aplica:
      - StandardScaler a columnas numéricas
      - OneHotEncoder a columnas categóricas
    """
    cat_cols = categorical_cols if categorical_cols is not None else CATEGORICAL_COLS
    num_cols = numeric_cols if numeric_cols is not None else NUMERIC_COLS

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return preprocessor


def split_features_target(df: pd.DataFrame):
    """
    Separa X (features) e y (target) a partir del DataFrame.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def clean_diabetes_dataset(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Limpia el dataset de Diabetes Health Indicators.

    Pasos:
    1. Estandariza strings en columnas categóricas.
    2. Convierte columnas binarias a 0/1.
    3. Winsoriza outliers en variables clínicas.
    4. (Opcional) Guarda el dataset limpio en CSV.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset original.
    save_path : str | Path, opcional
        Ruta donde guardar el CSV limpio.

    Retorna
    -------
    pd.DataFrame
        Dataset limpio.
    """

    # ==========================
    # 1. Estandarizar categóricas
    # ==========================
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()
        )


    binary_cols = [
        "family_history_diabetes",
        "hypertension_history",
        "cardiovascular_history",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace({"Yes": 1, "No": 0})
                .astype(float)
            )


    def winsorize(series: pd.Series, min_pct: float = 0.01, max_pct: float = 0.99):
        low = series.quantile(min_pct)
        high = series.quantile(max_pct)
        return np.clip(series, low, high)

    cols_outliers = [
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "glucose_fasting",
        "glucose_postprandial",
        "triglycerides",
        "hba1c",
        "insulin_level",
    ]

    for col in cols_outliers:
        if col in df.columns:
            df[col] = winsorize(df[col])


    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df
