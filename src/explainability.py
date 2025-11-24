import joblib
import shap
import pandas as pd

from .config import MODEL_PATH
from .data_loader import load_diabetes_data
from .preprocessing import split_features_target, build_preprocessor


def compute_shap_summary(sample_size: int = 2000):
    """
    Calcula valores SHAP sobre una muestra del dataset para interpretar el modelo.
    Nota: para RandomForest, se usa TreeExplainer; para logística, LinearExplainer.
    Como el modelo está envuelto en un Pipeline, usamos KernelExplainer como fallback.
    """
    df = load_diabetes_data()
    X, _ = split_features_target(df)

    if sample_size < len(X):
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    # Como el modelo es un Pipeline (preprocesamiento + clf),
    # es más seguro usar KernelExplainer sobre la salida predict_proba.
    # Para no explotar en tiempo, toma una submuestra pequeña como background.
    background = X_sample.sample(min(300, len(X_sample)), random_state=42)

    def model_proba_fn(data):
        return model.predict_proba(pd.DataFrame(data, columns=X_sample.columns))[:, 1]

    explainer = shap.KernelExplainer(model_proba_fn, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    shap.summary_plot(shap_values, X_sample)
