# Proyecto de Predicción de Diabetes

## Descripción del Proyecto

Este proyecto implementa un sistema completo de machine learning para la predicción de diabetes utilizando indicadores de salud. El modelo fue desarrollado como parte de un taller de clasificación y cuenta con una arquitectura modular, pipelines automatizados, análisis exploratorio, selección de modelos con validación cruzada, optimización de umbrales y una aplicación web interactiva.

## Objetivos

- Predecir el diagnóstico de diabetes a partir de variables clínicas y demográficas
- Comparar diferentes algoritmos de clasificación (Regresión Logística y Random Forest)
- Optimizar el umbral de decisión para maximizar el balance entre precisión y recall
- Implementar técnicas de balanceo de clases (SMOTE)
- Proporcionar explicabilidad mediante SHAP values
- Desplegar una aplicación interactiva con Streamlit

## Dataset

**Archivo:** `data/raw/diabetes_dataset.csv`

El dataset contiene información de pacientes con las siguientes características:

### Variables Demográficas
- `age`: Edad del paciente
- `gender`: Género (Male, Female, Other)
- `ethnicity`: Origen étnico
- `education_level`: Nivel educativo
- `income_level`: Nivel de ingresos
- `employment_status`: Estado laboral

### Variables de Estilo de Vida
- `alcohol_consumption_per_week`: Consumo de alcohol semanal
- `physical_activity_minutes_per_week`: Minutos de actividad física por semana
- `diet_score`: Puntuación de dieta
- `sleep_hours_per_day`: Horas de sueño diarias
- `screen_time_hours_per_day`: Horas de pantalla diarias
- `smoking_status`: Estado de fumador

### Variables Clínicas
- `bmi`: Índice de masa corporal
- `waist_to_hip_ratio`: Relación cintura-cadera
- `systolic_bp` / `diastolic_bp`: Presión arterial sistólica/diastólica
- `heart_rate`: Frecuencia cardíaca
- `cholesterol_total`, `hdl_cholesterol`, `ldl_cholesterol`, `triglycerides`: Perfiles de colesterol
- `glucose_fasting` / `glucose_postprandial`: Niveles de glucosa
- `insulin_level`: Nivel de insulina
- `hba1c`: Hemoglobina glicosilada
- `diabetes_risk_score`: Puntuación de riesgo de diabetes

### Variables de Historia Médica
- `family_history_diabetes`: Historia familiar de diabetes (0/1)
- `hypertension_history`: Historia de hipertensión (0/1)
- `cardiovascular_history`: Historia cardiovascular (0/1)

### Variable Objetivo
- `diagnosed_diabetes`: Diagnóstico de diabetes (0: No diabético, 1: Diabético)

## Estructura del Proyecto

```
MachineLearning1/
│
├── data/
│   ├── raw/                          # Datos originales
│   │   └── diabetes_dataset.csv
│   └── processed/                    # Datos procesados (generados)
│
├── src/                              # Código fuente modular
│   ├── config.py                     # Configuración y rutas
│   ├── data_loader.py                # Carga de datos
│   ├── preprocessing.py              # Preprocesamiento y transformaciones
│   ├── model_selection.py            # Comparación de modelos con CV
│   ├── train_model.py                # Entrenamiento y optimización
│   ├── evaluate.py                   # Evaluación del modelo
│   └── explainability.py             # Explicabilidad con SHAP
│
├── notebooks/                        # Notebooks de análisis
│   ├── eda_diabetes.ipynb           # Análisis exploratorio de datos
│   ├── modelado_cv_thresholds.ipynb # Modelado y optimización
│   └── shap_explicaciones.ipynb     # Análisis de explicabilidad
│
├── models/
│   └── modelo_diabetes.joblib        # Modelo entrenado (generado)
│
├── reports/
│   └── metrics.json                  # Métricas del modelo (generado)
│
├── app/
│   └── streamlit_app.py              # Aplicación web interactiva
│
├── main.py                           # Script principal CLI
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Documentación (este archivo)
```

## Instalación

### Prerrequisitos
- Python 3.8 o superior
- pip

### Pasos de Instalación

1. **Clonar o descargar el proyecto**

2. **Crear un entorno virtual (recomendado)**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Dependencias Principales
```
pandas               # Manipulación de datos
numpy                # Operaciones numéricas
scikit-learn         # Algoritmos de ML y métricas
matplotlib           # Visualización
joblib               # Serialización de modelos
streamlit            # Aplicación web
imbalanced-learn     # Técnicas de balanceo (SMOTE)
shap                 # Explicabilidad del modelo
```

## Uso del Proyecto

### 1. Entrenamiento del Modelo

Para entrenar el modelo desde cero:

```bash
python main.py train
```

**¿Qué hace este comando?**
- Carga y limpia el dataset
- Divide los datos en train/test (80/20) de forma estratificada
- Compara múltiples modelos con validación cruzada 5-fold:
  - Regresión Logística (con y sin SMOTE)
  - Random Forest (con y sin SMOTE)
- Selecciona el mejor modelo según ROC-AUC
- Calibra las probabilidades usando Platt Scaling
- Optimiza el umbral de decisión para maximizar F1-score (min. precisión 60%)
- Guarda el modelo en `models/modelo_diabetes.joblib`
- Guarda las métricas en `reports/metrics.json`

### 2. Evaluación del Modelo

Para evaluar el modelo guardado:

```bash
python main.py evaluate
```

**¿Qué hace este comando?**
- Carga el modelo entrenado
- Evalúa sobre el dataset completo
- Muestra métricas detalladas:
  - ROC-AUC
  - Balanced Accuracy
  - Classification Report (precision, recall, F1-score)

### 3. Aplicación Web Interactiva

Para iniciar la aplicación de predicción:

```bash
streamlit run app/streamlit_app.py
```

**Características de la app:**
- Interfaz amigable para introducir datos de pacientes
- Predicción en tiempo real
- Visualización de probabilidades
- Información del modelo (ROC-AUC, umbral óptimo)

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### 4. Notebooks de Análisis

Los notebooks están ubicados en `notebooks/` y pueden ejecutarse en Jupyter/DataSpell:

- **`eda_diabetes.ipynb`**: Análisis exploratorio completo
  - Estadísticas descriptivas
  - Visualización de distribuciones
  - Análisis de correlaciones
  - Detección de outliers
  - Balance de clases

- **`modelado_cv_thresholds.ipynb`**: Experimentación con modelos
  - Comparación de algoritmos
  - Validación cruzada
  - Curvas ROC y Precision-Recall
  - Optimización de umbrales
  - Métricas de desempeño

- **`shap_explicaciones.ipynb`**: Interpretabilidad del modelo
  - Cálculo de SHAP values
  - Feature importance
  - Análisis de contribuciones individuales

## Resultados del Modelo

### Métricas de Desempeño

Según el archivo `reports/metrics.json`, el modelo entrenado alcanza:

| Métrica | Valor |
|---------|-------|
| **Modelo Seleccionado** | Random Forest |
| **ROC-AUC (CV)** | 0.9999 |
| **ROC-AUC (Test)** | 0.9999 |
| **Balanced Accuracy** | 0.9997 |
| **Umbral Óptimo** | 0.602 |
| **Precisión** | 1.000 |
| **Recall** | 0.9995 |

> **Nota:** Estos resultados excepcionalmente altos sugieren que el dataset puede ser sintético o altamente separable. En aplicaciones reales, es importante validar con datos externos y considerar posible overfitting.

### Pipeline de Procesamiento

El modelo implementa el siguiente pipeline:

1. **Preprocesamiento:**
   - StandardScaler para variables numéricas
   - OneHotEncoder para variables categóricas

2. **Balanceo (opcional):**
   - SMOTE si el desbalance de clases es significativo

3. **Modelo:**
   - Random Forest con 200 árboles
   - Class weight: balanced_subsample

4. **Calibración:**
   - CalibratedClassifierCV con método sigmoid (Platt Scaling)

5. **Optimización de Umbral:**
   - Búsqueda del umbral que maximiza F1-score
   - Restricción de precisión mínima del 60%

## Explicabilidad

El proyecto incluye análisis de explicabilidad usando **SHAP (SHapley Additive exPlanations)**:

```python
from src.explainability import compute_shap_summary

# Genera gráficos de feature importance
compute_shap_summary(sample_size=2000)
```

Esto permite entender:
- ¿Qué variables contribuyen más a las predicciones?
- ¿Cómo afecta cada característica al riesgo de diabetes?
- Interpretación de predicciones individuales

## Módulos del Código

### `config.py`
Define rutas y configuraciones globales del proyecto.

### `data_loader.py`
```python
load_diabetes_data(path=None, clean=False, save_clean_path=None)
```
Carga el dataset con opción de aplicar limpieza.

### `preprocessing.py`
- `build_preprocessor()`: Crea el ColumnTransformer
- `split_features_target()`: Separa X e y
- `clean_diabetes_dataset()`: Limpia y estandariza datos

### `model_selection.py`
```python
compare_models(X, y, use_smote=False)
```
Compara modelos con validación cruzada estratificada y retorna resultados.

### `train_model.py`
```python
train_and_save_model(test_size=0.2, random_state=42)
```
Pipeline completo de entrenamiento, calibración y guardado.

### `evaluate.py`
```python
evaluate_saved_model()
```
Evalúa el modelo guardado sobre el dataset.

### `explainability.py`
```python
compute_shap_summary(sample_size=2000)
```
Calcula y visualiza SHAP values para interpretabilidad.

## Personalización

### Cambiar el modelo
Edita `src/model_selection.py` para agregar nuevos modelos a la comparación:

```python
def _build_nuevo_modelo_pipeline(use_smote=False):
    preprocessor = build_preprocessor()
    clf = TuModelo(...)
    # ... construir pipeline
    return pipeline
```

### Ajustar el umbral de decisión
Modifica el parámetro `min_precision` en `train_model.py`:

```python
thr_opt, prec_opt, rec_opt = _find_best_threshold(
    y_test, y_proba, 
    min_precision=0.7  # Cambiar según necesidades
)
```

### Agregar nuevas features a la app
Edita `app/streamlit_app.py` para incluir más campos de entrada según tus necesidades.

## Mejores Prácticas Implementadas

- **Arquitectura modular** con separación de responsabilidades
- **Configuración centralizada** en `config.py`
- **Validación cruzada estratificada** para evaluación robusta
- **Calibración de probabilidades** para predicciones confiables
- **Optimización de umbrales** considerando trade-offs negocio/técnicos
- **Manejo de desbalance** con class_weight y SMOTE
- **Reproducibilidad** con random_state fijado
- **Explicabilidad** mediante SHAP values
- **Interfaz de usuario** intuitiva con Streamlit

## Limitaciones y Consideraciones

1. **Dataset Sintético:** Los resultados perfectos sugieren que el dataset puede ser sintético. En producción, validar con datos reales.

2. **Overfitting:** Con ROC-AUC > 0.999, existe riesgo de overfitting. Validar con datos externos.

3. **Aplicación Médica:** Este modelo es solo para fines académicos/demostrativos. No debe usarse para diagnósticos médicos reales sin validación clínica apropiada.

4. **Escalabilidad:** Para datasets grandes, considerar técnicas de muestreo o modelos más eficientes.

## Referencias

- **Scikit-learn:** https://scikit-learn.org/
- **SMOTE:** https://imbalanced-learn.org/
- **SHAP:** https://github.com/slundberg/shap
- **Streamlit:** https://streamlit.io/

## Autor

Proyecto desarrollado como parte del Taller de Clasificación - Machine Learning 1

## Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

---

**¿Cómo leer el archivo `diabetes_dataset.csv`?**

Para leer el dataset en Python:

```python
import pandas as pd

# Opción 1: Ruta relativa
df = pd.read_csv('data/raw/diabetes_dataset.csv')

# Opción 2: Usando el módulo del proyecto
from src.data_loader import load_diabetes_data
df = load_diabetes_data()

# Con limpieza aplicada
df_clean = load_diabetes_data(clean=True)
```

