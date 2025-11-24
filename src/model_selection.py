import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .preprocessing import build_preprocessor


@dataclass
class ModelCVResult:
    name: str
    estimator: Any
    cv_scores: Dict[str, float]


def _build_logistic_pipeline(use_smote: bool = False):
    preprocessor = build_preprocessor()
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )

    if use_smote:
        pipeline = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", clf),
            ]
        )
    else:
        pipeline = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", clf),
            ]
        )
    return pipeline


def _build_rf_pipeline(use_smote: bool = False):
    preprocessor = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    if use_smote:
        pipeline = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", clf),
            ]
        )
    else:
        pipeline = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", clf),
            ]
        )
    return pipeline


def compare_models(X, y, use_smote: bool = False) -> Dict[str, ModelCVResult]:
    """
    Compara modelos con validación cruzada estratificada.
    Devuelve un dict con resultados promedio de métricas.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "roc_auc": "roc_auc",
        "bal_acc": "balanced_accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "pr_auc": "average_precision",
    }

    models = {
        "logistic": _build_logistic_pipeline(use_smote=use_smote),
        "random_forest": _build_rf_pipeline(use_smote=use_smote),
    }

    results: Dict[str, ModelCVResult] = {}

    for name, estimator in models.items():
        cv_res = cross_validate(
            estimator,
            X,
            y,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        mean_scores = {metric: float(np.mean(cv_res[f"test_{metric}"]))
                       for metric in scoring.keys()}

        results[name] = ModelCVResult(
            name=name,
            estimator=estimator,
            cv_scores=mean_scores,
        )

    return results
