import json
from pathlib import Path

import streamlit as st
import joblib
import pandas as pd

from src.config import MODEL_PATH, METRICS_PATH, NUMERIC_COLS, CATEGORICAL_COLS

st.set_page_config(page_title="Predicción de Diabetes", page_icon="🩺")

st.title("🩺 Predicción de Diabetes – Indicadores de Salud")

# Cargar modelo
if not MODEL_PATH.exists():
    st.error("No se encontró el modelo entrenado. Ejecuta primero `python main.py train`.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
thr_opt = bundle["optimal_threshold"]

metrics = {}
if METRICS_PATH.exists():
    metrics = json.loads(Path(METRICS_PATH).read_text(encoding="utf-8"))

st.sidebar.header("Información del modelo")
if metrics:
    st.sidebar.write(f"**Modelo:** {metrics.get('best_model_name')}")
    st.sidebar.write(f"**ROC-AUC test:** {metrics.get('test_roc_auc'):.3f}")
    st.sidebar.write(f"**Umbral óptimo:** {metrics.get('optimal_threshold'):.3f}")
else:
    st.sidebar.write("Métricas no disponibles aún.")

st.markdown("Introduce los datos del/la paciente:")

# Para demo: solo algunos campos principales (puedes ampliarlo)
age = st.number_input("Edad", min_value=18, max_value=100, value=50)
gender = st.selectbox("Género", ["Male", "Female", "Other"])
ethnicity = st.selectbox("Etnia", ["White", "Black", "Asian", "Hispanic", "Other"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0)
waist_to_hip_ratio = st.number_input("Relación cintura/cadera", min_value=0.5, max_value=2.0, value=0.9)
systolic_bp = st.number_input("Presión sistólica", min_value=80, max_value=220, value=120)
diastolic_bp = st.number_input("Presión diastólica", min_value=40, max_value=130, value=80)
glucose_fasting = st.number_input("Glucosa en ayunas", min_value=60, max_value=400, value=110)
hba1c = st.number_input("HbA1c", min_value=4.0, max_value=15.0, value=6.0)
family_history_diabetes = st.selectbox("Historia familiar de diabetes", [0, 1])

# Construir un DataFrame con TODAS las columnas esperadas por el modelo
input_data = {col: [None] for col in NUMERIC_COLS + CATEGORICAL_COLS}

# Rellenar numéricas básicas
input_data["age"] = [age]
input_data["bmi"] = [bmi]
input_data["waist_to_hip_ratio"] = [waist_to_hip_ratio]
input_data["systolic_bp"] = [systolic_bp]
input_data["diastolic_bp"] = [diastolic_bp]
input_data["glucose_fasting"] = [glucose_fasting]
input_data["hba1c"] = [hba1c]
input_data["family_history_diabetes"] = [family_history_diabetes]

# Rellenar algunas categóricas
input_data["gender"] = [gender]
input_data["ethnicity"] = [ethnicity]
# Otras categóricas puedes poner defaults, e.g.:
input_data["education_level"] = ["Highschool"]
input_data["income_level"] = ["Middle"]
input_data["employment_status"] = ["Employed"]
input_data["smoking_status"] = ["Never"]
input_data["diabetes_stage"] = ["No Diabetes"]

X_new = pd.DataFrame(input_data)

if st.button("Predecir riesgo de diabetes"):
    proba = model.predict_proba(X_new)[:, 1][0]
    pred = int(proba >= thr_opt)

    st.markdown(f"**Probabilidad estimada de diabetes:** {proba:.3f}")
    st.markdown(f"**Clasificación (umbral {thr_opt:.2f}):** {'Diabético' if pred == 1 else 'No diabético'}")

    st.info("Recuerda: este modelo es un ejemplo académico y no reemplaza una evaluación médica profesional.")
