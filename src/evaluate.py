import json
import joblib
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report

from .config import MODEL_PATH, METRICS_PATH
from .data_loader import load_diabetes_data
from .preprocessing import split_features_target


def evaluate_saved_model():
    """
    Carga el modelo guardado y lo evalúa de nuevo sobre todo el dataset
    (o sobre un subconjunto si se quiere).
    """
    df = load_diabetes_data()
    X, y = split_features_target(df)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    thr = bundle["optimal_threshold"]

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= thr).astype(int)

    roc = roc_auc_score(y, y_proba)
    bal_acc = balanced_accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, digits=3)

    print("=== Evaluación global sobre el dataset completo ===")
    print(f"ROC-AUC: {roc:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print("\nClassification report:")
    print(report)

    # Guardar resumen extendido
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metrics.update({
        "full_data_roc_auc": roc,
        "full_data_balanced_accuracy": bal_acc,
    })

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
