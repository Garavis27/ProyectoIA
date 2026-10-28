# Actualización: entrenamiento con CSV y resultados por paciente

Ejecute desde esta carpeta:

```bash
python pipeline_eda_y_entrenamiento_grupo01.py
```

El programa lee `dataset_pacientes_cronicos_grupo01.csv` de la carpeta principal. Para usar otro archivo con las mismas columnas: `python pipeline_eda_y_entrenamiento_grupo01.py --dataset "ruta/al/archivo.csv"`. No modifica el dataset de entrada. Exige identificadores únicos, campos requeridos sin valores vacíos y ambas clases (0 y 1).

El archivo `resultados/dataset/predicciones_pacientes_prueba.csv` contiene las características originales y, para cada uno de los tres modelos, el puntaje, umbral, predicción y resultado individual (VP, FN, FP o VN). Incluye únicamente los pacientes del conjunto de prueba, no los de entrenamiento ni validación. Abra el CSV en Excel y filtre `Resultado_Gradient_Boosting`: **VP - Identificado** o **FN - No identificado**. El puntaje es una salida del modelo y no debe interpretarse como probabilidad clínica validada.

Las métricas y figuras se regeneran en `resultados/figuras_y_metricas/`. La división se hace con semilla fija; los valores actuales pueden diferir de los documentados históricamente abajo. El modelo llamado XGBoost en los entregables anteriores es una implementación propia de gradient boosting, no la biblioteca XGBoost. Los datos incluidos son simulados y no demuestran desempeño clínico real. Los textos y cifras del documento histórico de abajo no reflejan necesariamente la ejecución actual; las rutas `src/` y `notebooks/` descritas allí no corresponden a los archivos presentes.

---

# Sistema Inteligente de Predicción Temprana de Complicaciones Clínicas a 6–12 Meses en Pacientes Crónicos (Diabetes Mellitus, Hipertensión Arterial y Enfermedad Renal Crónica)

**Asignatura:** Inteligencia Artificial — Ingeniería de Sistemas (Séptimo Semestre)  
**Equipo de Investigación — Grupo 01:** Jorlan Rangel, Alex Garavis y Juanma García  
**Alcance del Repositorio:** Entregables de la **Fase 1 (Formulación y Exploración — Semanas 1–2)** y **Fase 2 (Desarrollo y Modelado — Semanas 3–7)**, abarcando desde la **Fase 01 hasta la Fase 10 del Ciclo de Vida de Entrenamiento de Machine Learning**.

## 1. Descripción del Proyecto y Mapeo de las 10 Fases de Machine Learning

Este repositorio implementa una solución completa de aprendizaje automático supervisado para estratificar el riesgo individual de desarrollar complicaciones agudas o progresión severa a 6–12 meses en pacientes diagnosticados con Diabetes Mellitus Tipo 2 (DM), Hipertensión Arterial (HTA) y Enfermedad Renal Crónica (ERC estadios 1 a 4). El flujo computacional integra registros del dataset público [CDC Diabetes Health Indicators (BRFSS)](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) disponible en [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), estadísticas epidemiológicas de la [Cuenta de Alto Costo de Colombia](https://cuentadealtocosto.org/noticias/dia-mundial-del-corazon-2026/) y variables clínicas longitudinales extraídas de RIPS e historias clínicas electrónicas simuladas ($N = 10,000$ pacientes limpios).

| Fase del Proyecto de Aula | Fase de Entrenamiento de IA (ML Lifecycle) | Módulo / Archivo en el Repositorio | Estado |
| :--- | :--- | :--- | :--- |
| **Fase 1** (Semanas 1–2) | **Fase 01:** Definir el problema | `docs/Proyecto_Aula_IA_Grupo01_Fases_1_y_2.pdf` (Sección 1) | Completado |
| **Fase 1** (Semanas 1–2) | **Fase 02:** Recolección de datos | `data/dataset_pacientes_cronicos_grupo01.csv` | Completado |
| **Fase 1** (Semanas 1–2) | **Fase 03:** Exploración y análisis de datos (EDA) | `notebooks/Notebook_EDA_y_Modelado_Grupo01_Diabetes.ipynb` | Completado |
| **Fase 2** (Semanas 3–7) | **Fase 04:** Limpieza y preparación de datos | `src/preprocessing.py` & Notebook (Sección Fase 04) | Completado |
| **Fase 2** (Semanas 3–7) | **Fase 05:** Ingeniería de características | `src/feature_engineering.py` & Notebook (Sección Fase 05) | Completado |
| **Fase 2** (Semanas 3–7) | **Fase 06:** Dividir los datos (70% / 15% / 15%) | `src/split_data.py` & Notebook (Sección Fase 06) | Completado |
| **Fase 2** (Semanas 3–7) | **Fase 07:** Seleccionar el algoritmo | `src/models.py` (Regresión Logística, Random Forest, XGBoost) | Completado |
| **Fase 2** (Semanas 3–7) | **Fase 08:** Entrenar el modelo | `src/train.py` & Notebook (Sección Fase 08) | Completado |
| **Fase 2** (Semanas 3–7) | **Fase 09:** Ajustar hiperparámetros | `src/tuning.py` & Notebook (Sección Fase 09) | Completado |
| **Fase 2** (Semanas 3–7) | **Fase 10:** Evaluar el modelo (Matriz de Confusión y ROC-AUC) | `src/evaluate.py` & Notebook (Sección Fase 10) | Completado |
| *Fase 3 (Semanas 8–12)* | *Fase 11 y 12: Generalización y Despliegue (Streamlit / API)* | `app/streamlit_dashboard.py` *(Próxima Entrega)* | En progreso |

## 2. Arquitectura del Repositorio

```text
proyecto-ia-grupo01-diabetes/
├── README.md                                                # Documentación principal e instrucciones de reproducción
├── requirements.txt                                         # Dependencias de Python (numpy, pandas, scipy, matplotlib, seaborn)
├── data/
│   └── dataset_pacientes_cronicos_grupo01.csv               # Dataset integrado limpio (10,000 pacientes x 23 variables)
├── notebooks/
│   └── Notebook_EDA_y_Modelado_Grupo01_Diabetes.ipynb       # Notebook ejecutable con EDA (Fase 1) y Modelado (Fase 2 / Fases ML 1-10)
├── docs/
│   ├── Proyecto_Aula_IA_Grupo01_Fases_1_y_2.pdf             # Documento Técnico Consolidado (Propuesta + Avance Fase 2)
│   └── Proyecto_Aula_IA_Grupo01_Fases_1_y_2.docx            # Versión editable en Microsoft Word
└── src/
    └── run_ml_pipeline.py                                   # Pipeline reproducible completo de las Fases 01 a 10 de ML
```

## 3. Resultados Principales en el Conjunto de Prueba ($N = 1,500$ pacientes)

| Modelo Evaluado | Verdaderos Positivos (VP) | Falsos Negativos (FN) | Falsos Positivos (FP) | Verdaderos Negativos (VN) | Accuracy | Precision | Recall (Sensibilidad) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Modelo Ingenuo** (Predice siempre `0`) | 0 | 248 | 0 | 1,252 | 83.47% | 0.00% | 0.00% | 0.00% | 0.5000 |
| **Regresión Logística** (Baseline Lineal) | 183 | 65 | 90 | 1,162 | 89.67% | 67.03% | 73.79% | 70.25% | 0.9189 |
| **Random Forest** (Ensamble Bagging) | 183 | 65 | 78 | 1,174 | 90.47% | 70.11% | **73.79%** | 71.91% | **0.9265** |
| **XGBoost Classifier** (Ensamble Boosting) | 180 | 68 | 72 | 1,180 | **90.67%** | **71.43%** | 72.58% | **72.00%** | 0.9238 |

## 4. Evidencia de Control de Versiones Incremental (Git Workflow Semanas 1–7)

Para cumplir con el criterio de evaluación de **trabajo incremental distribuido en el tiempo (15% del peso de la Fase 2)**, el equipo estructuró tres ramas de trabajo (`main`, `develop` y ramas por componente funcional) registrando commits semanales verificables:

| Semana | Fecha | Autor | Rama | Hash / Mensaje del Commit (Convención Semántica) | Fase ML Cubierta |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **Sem 1** | `2026-09-08` | Jorlan Rangel | `main` | `docs(propuesta): definir planteamiento del problema, variable objetivo y estado del arte` | **Fase 01 ML** |
| **Sem 1** | `2026-09-12` | Alex Garavis | `feat/data` | `feat(datos): integrar fuentes BRFSS Kaggle, estructura RIPS e historias clinicas (N=10120)` | **Fase 02 ML** |
| **Sem 2** | `2026-09-18` | Juanma García | `feat/eda` | `feat(eda): ejecutar analisis exploratorio, distribuciones, deteccion de nulos y desbalance` | **Fase 03 ML** |
| **Sem 2** | `2026-09-21` | Jorlan Rangel | `main` | `docs(cronograma): incorporar diagrama de Gantt semanas 1-12 y cierre de Entrega Fase 1` | **Cierre Fase 1 Aula** |
| **Sem 3** | `2026-09-29` | Alex Garavis | `feat/prep` | `fix(limpieza): eliminar 120 duplicados RIPS, truncar outliers de digitacion e imputar medianas` | **Fase 04 ML** |
| **Sem 4** | `2026-10-06` | Juanma García | `feat/fe` | `feat(features): crear Indice_Riesgo_CardioRenal, Presion_Pulso y sinergias gluco-renales` | **Fase 05 ML** |
| **Sem 5** | `2026-10-13` | Jorlan Rangel | `feat/split` | `feat(particion): implementar split estratificado 70/15/15 (7000 train, 1500 val, 1500 test)` | **Fase 06 ML** |
| **Sem 5** | `2026-10-16` | Alex Garavis | `feat/models` | `feat(modelos): implementar Regresion Logistica L2 (baseline), Random Forest y XGBoost` | **Fase 07 y 08 ML** |
| **Sem 6** | `2026-10-21` | Juanma García | `feat/tuning` | `perf(tuning): ejecutar busqueda de hiperparametros y calibracion de umbral en validacion` | **Fase 09 ML** |
| **Sem 7** | `2026-10-26` | Jorlan Rangel | `main` | `eval(fase10): generar matrices de confusion, curvas ROC-AUC y documento tecnico Fase 2` | **Fase 10 ML** |

## 5. Instrucciones de Instalación y Ejecución Local

```bash
# 1. Clonar el repositorio y ubicarse en la carpeta raíz
git clone https://github.com/grupo01-ia-sistemas/prediccion-complicaciones-diabetes-erc.git
cd prediccion-complicaciones-diabetes-erc

# 2. Ejecutar el pipeline de preprocesamiento, entrenamiento y evaluación (Fases 01 a 10 ML)
python3 src/run_ml_pipeline.py

# 3. Abrir el Notebook interactivo en Jupyter Lab o Google Colab
jupyter notebook notebooks/Notebook_EDA_y_Modelado_Grupo01_Diabetes.ipynb
```
