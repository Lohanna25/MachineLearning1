from pathlib import Path

# Rutas base
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "diabetes_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_diabetes.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

TARGET_COL = "diagnosed_diabetes"

# Columnas (puedes ajustar si cambian los nombres)
CATEGORICAL_COLS = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status",
    "diabetes_stage",
]

# Todas menos target y categóricas:
NUMERIC_COLS = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
    "diabetes_risk_score",
]
