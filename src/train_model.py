import json
from typing import Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    precision_recall_curve,
    classification_report,
)

from .config import MODEL_PATH, METRICS_PATH
from .data_loader import load_diabetes_data
from .preprocessing import split_features_target
from .model_selection import compare_models


def _select_best_model(cv_results):
    """
    Elige el mejor modelo según ROC-AUC (podrías cambiar la lógica).
    """
    best_name = None
    best_score = -np.inf
    best_estimator = None

    for name, res in cv_results.items():
        score = res.cv_scores["roc_auc"]
        if score > best_score:
            best_score = score
            best_name = name
            best_estimator = res.estimator

    return best_name, best_estimator, best_score


def _find_best_threshold(y_true, y_proba, min_precision: float = 0.6) -> Tuple[float, float, float]:
    """
    Busca un umbral que mantenga al menos cierta precisión
    y maximice el F1-score (buen equilibrio entre precision y recall).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    best_thr = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for t, p, r in zip(thresholds, precisions[1:], recalls[1:]):
        if p < min_precision:
            continue  # descartamos umbrales con precision menor a la mínima

        f1 = 2 * p * r / (p + r + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
            best_precision = p
            best_recall = r

    return float(best_thr), float(best_precision), float(best_recall)



def train_and_save_model(test_size: float = 0.2, random_state: int = 42):
    """
    Entrena el modelo, calibra probabilidades y guarda modelo + métricas.
    """
    df = load_diabetes_data(clean=True)
    X, y = split_features_target(df)

    # Train/test split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Determinar si usamos SMOTE (ejemplo: si minoritaria < 0.3)
    pos_rate = y.mean()
    use_smote = pos_rate < 0.3 or pos_rate > 0.7

    # Comparar modelos
    cv_results = compare_models(X_train, y_train, use_smote=use_smote)
    best_name, best_estimator, best_cv_roc_auc = _select_best_model(cv_results)

    print(f"Mejor modelo según ROC-AUC CV: {best_name} ({best_cv_roc_auc:.3f})")

    # Calibrar probabilidades (Platt)
    calibrated_clf = CalibratedClassifierCV(
        estimator=best_estimator,
        cv=3,
        method="sigmoid",
    )

    calibrated_clf.fit(X_train, y_train)

    # Evaluación final en test
    y_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    bal_acc_default = balanced_accuracy_score(y_test, (y_proba >= 0.5).astype(int))

    thr_opt, prec_opt, rec_opt = _find_best_threshold(y_test, y_proba, min_precision=0.6)
    y_pred_opt = (y_proba >= thr_opt).astype(int)
    bal_acc_opt = balanced_accuracy_score(y_test, y_pred_opt)

    print("\n=== Clasification report (umbral óptimo) ===")
    print(classification_report(y_test, y_pred_opt, digits=3))

    metrics = {
        "best_model_name": best_name,
        "cv_roc_auc": best_cv_roc_auc,
        "test_roc_auc": roc_auc,
        "test_balanced_accuracy_default_thr": bal_acc_default,
        "optimal_threshold": thr_opt,
        "optimal_precision": prec_opt,
        "optimal_recall": rec_opt,
        "test_balanced_accuracy_opt_thr": bal_acc_opt,
    }

    # Guardar modelo y métricas
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": calibrated_clf,
        "optimal_threshold": thr_opt,
    }, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")

    return metrics
