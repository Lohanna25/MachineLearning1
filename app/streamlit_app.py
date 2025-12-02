import json
from pathlib import Path

import streamlit as st
import joblib
import pandas as pd

import sys
import os

# Agregar carpeta raíz del proyecto al PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


from src.config import MODEL_PATH, METRICS_PATH, NUMERIC_COLS, CATEGORICAL_COLS

# ---------------------------------------------------
# Configuración básica de la página
# ---------------------------------------------------
st.set_page_config(page_title="Predicción de Diabetes", page_icon="🩺")

st.title("🩺 Predicción de Diabetes – Indicadores de salud (sin exámenes de laboratorio)")
st.write(
    """
    Este modelo estima el riesgo de que una persona tenga diabetes 
    usando solo información demográfica, hábitos de vida, antecedentes 
    y signos vitales. No utiliza resultados de laboratorio.
    """
)

# ---------------------------------------------------
# Cargar modelo y métricas
# ---------------------------------------------------
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

# ---------------------------------------------------
# Formulario de entrada
# ---------------------------------------------------
st.markdown("### Datos del/la paciente")

# NUMERIC_COLS esperadas por el modelo:
# [
#   "age",
#   "alcohol_consumption_per_week",
#   "physical_activity_minutes_per_week",
#   "diet_score",
#   "sleep_hours_per_day",
#   "screen_time_hours_per_day",
#   "family_history_diabetes",
#   "hypertension_history",
#   "cardiovascular_history",
#   "bmi",
#   "waist_to_hip_ratio",
#   "systolic_bp",
#   "diastolic_bp",
#   "heart_rate",
# ]

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", min_value=18, max_value=100, value=50)
    alcohol = st.number_input(
        "Consumo de alcohol por semana (unidades)",
        min_value=0.0,
        max_value=60.0,
        value=1.0,
        step=0.5,
    )
    pa_minutes = st.number_input(
        "Minutos de actividad física por semana",
        min_value=0,
        max_value=2000,
        value=150,
        step=10,
    )
    diet_score = st.number_input(
        "Puntaje de dieta (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
    )
    sleep_hours = st.number_input(
        "Horas de sueño por día",
        min_value=3.0,
        max_value=14.0,
        value=7.0,
        step=0.5,
    )
    screen_time = st.number_input(
        "Horas frente a pantallas por día",
        min_value=0.0,
        max_value=18.0,
        value=4.0,
        step=0.5,
    )
    bmi = st.number_input(
        "Índice de masa corporal (BMI)",
        min_value=10.0,
        max_value=60.0,
        value=28.0,
        step=0.1,
    )

with col2:
    waist_to_hip_ratio = st.number_input(
        "Relación cintura/cadera",
        min_value=0.5,
        max_value=2.0,
        value=0.9,
        step=0.01,
    )
    systolic_bp = st.number_input(
        "Presión sistólica",
        min_value=80,
        max_value=260,
        value=120,
        step=1,
    )
    diastolic_bp = st.number_input(
        "Presión diastólica",
        min_value=40,
        max_value=150,
        value=80,
        step=1,
    )
    heart_rate = st.number_input(
        "Frecuencia cardiaca (latidos por minuto)",
        min_value=40,
        max_value=200,
        value=75,
        step=1,
    )

    family_history_diabetes = st.selectbox(
        "Historia familiar de diabetes",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí",
    )
    hypertension_history = st.selectbox(
        "Historia de hipertensión",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí",
    )
    cardiovascular_history = st.selectbox(
        "Historia cardiovascular",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Sí",
    )

st.markdown("### Características demográficas y hábitos")

# CATEGORICAL_COLS esperadas:
# [
#   "gender",
#   "ethnicity",
#   "education_level",
#   "income_level",
#   "employment_status",
#   "smoking_status",
# ]

gender = st.selectbox("Género", ["Male", "Female", "Other"])
ethnicity = st.selectbox("Etnia", ["White", "Black", "Asian", "Hispanic", "Other"])
education_level = st.selectbox(
    "Nivel educativo",
    ["Less Than Highschool", "Highschool", "Undergraduate", "Graduate"],
)
income_level = st.selectbox(
    "Nivel de ingresos",
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
)
employment_status = st.selectbox(
    "Situación laboral",
    ["Employed", "Unemployed", "Retired", "Student", "Other"],
)
smoking_status = st.selectbox(
    "Estado de tabaquismo",
    ["Never", "Former", "Current"],
)

# ---------------------------------------------------
# Construir DataFrame exactamente con las columnas
# que el modelo espera (NUMERIC_COLS + CATEGORICAL_COLS)
# ---------------------------------------------------
input_data = {
    # Numéricas
    "age": [age],
    "alcohol_consumption_per_week": [alcohol],
    "physical_activity_minutes_per_week": [pa_minutes],
    "diet_score": [diet_score],
    "sleep_hours_per_day": [sleep_hours],
    "screen_time_hours_per_day": [screen_time],
    "family_history_diabetes": [int(family_history_diabetes)],
    "hypertension_history": [int(hypertension_history)],
    "cardiovascular_history": [int(cardiovascular_history)],
    "bmi": [bmi],
    "waist_to_hip_ratio": [waist_to_hip_ratio],
    "systolic_bp": [systolic_bp],
    "diastolic_bp": [diastolic_bp],
    "heart_rate": [heart_rate],

    # Categóricas
    "gender": [gender],
    "ethnicity": [ethnicity],
    "education_level": [education_level],
    "income_level": [income_level],
    "employment_status": [employment_status],
    "smoking_status": [smoking_status],
}

X_new = pd.DataFrame(input_data)

# ---------------------------------------------------
# Predicción
# ---------------------------------------------------
if st.button("Predecir riesgo de diabetes"):
    proba = model.predict_proba(X_new)[:, 1][0]
    pred = int(proba >= thr_opt)

    st.markdown(f"### Resultado")
    st.markdown(f"**Probabilidad estimada de diabetes:** {proba:.3f}")
    st.markdown(
        f"**Clasificación (umbral {thr_opt:.2f}):** "
        f"{'Paciente con ALTO RIESGO de diabetes' if pred == 1 else 'Paciente con BAJO RIESGO de diabetes'}"
    )


