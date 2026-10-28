#!/usr/bin/env python3
"""
Script de Ejecución Real del Ciclo de Vida de Machine Learning (Fases 1 a 10)
Grupo 01: Jorlan Rangel, Alex Garavis, Juanma García
Proyecto: Predicción de complicaciones en pacientes crónicos (Diabetes, HTA, ERC) a 6-12 meses.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import expit

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "resultados", "figuras_y_metricas")
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados", "dataset")
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Los resultados se guardarán en: {os.path.join(BASE_DIR, 'resultados')}")

np.random.seed(42)

# ==============================================================================
# FASE 2 ML: CARGA DEL DATASET EXISTENTE
# ==============================================================================
parser = argparse.ArgumentParser(description="Entrenamiento y reporte individual de pacientes")
parser.add_argument("--dataset", default=os.path.join(BASE_DIR, "dataset_pacientes_cronicos_grupo01.csv"),
                    help="Ruta del archivo CSV de entrada")
args = parser.parse_args()
if not os.path.isfile(args.dataset):
    raise FileNotFoundError(f"No se encontró el dataset: {args.dataset}")
df_raw = pd.read_csv(args.dataset)
required = ['ID_Paciente', 'Edad', 'Sexo', 'Zona_Residencia', 'Diabetes_Mellitus',
            'Hipertension_Arterial', 'Estadio_ERC_Base', 'IMC', 'HbA1c_Porcentaje',
            'PAS_mmHg', 'PAD_mmHg', 'TFG_mL_min', 'Creatinina_mg_dL',
            'Albuminuria_ACR_mg_g', 'Colesterol_Alto', 'Tabaquismo', 'Actividad_Fisica',
            'Adherencia_Farmacologica', 'Consultas_Control_12m', 'Urgencias_Previas_12m',
            'Polifarmacia_Num_Meds', 'Salud_General_BRFSS', 'Complicacion_6_12m']
missing = sorted(set(required) - set(df_raw.columns))
if missing:
    raise ValueError(f"Faltan columnas en el dataset: {', '.join(missing)}")
if df_raw[required].isna().any().any():
    raise ValueError("El dataset contiene valores vacíos en columnas requeridas; límpielos antes de entrenar.")
if df_raw['ID_Paciente'].duplicated().any():
    raise ValueError("El dataset contiene ID_Paciente duplicados; cada paciente debe tener un ID único.")
if set(df_raw['Complicacion_6_12m'].unique()) != {0, 1}:
    raise ValueError("Complicacion_6_12m debe contener ambas clases: 0 y 1.")
for col, valid in {'Sexo': {'Masculino', 'Femenino'}, 'Zona_Residencia': {'Urbana', 'Rural'},
                   'Adherencia_Farmacologica': {'Baja', 'Media', 'Alta'}}.items():
    if not set(df_raw[col].unique()).issubset(valid):
        raise ValueError(f"Valores no reconocidos en {col}. Valores permitidos: {sorted(valid)}")
df_clean_base = df_raw.copy().reset_index(drop=True)
print(f"[1/6] Dataset cargado: {args.dataset} ({len(df_clean_base)} pacientes).")

# ==============================================================================
# FASE 3 ML: EXPLORACIÓN Y ANÁLISIS DE DATOS (EDA)
# ==============================================================================
eda_stats = {
    'n_raw': int(len(df_raw)),
    'n_duplicates': int(df_raw.duplicated().sum()),
    'n_clean': int(len(df_clean_base)),
    'prevalence_pos': round(float(df_clean_base['Complicacion_6_12m'].mean() * 100), 2),
    'prevalence_neg': round(float((1 - df_clean_base['Complicacion_6_12m'].mean()) * 100), 2),
    'nulls_hba1c_pct': round(float(df_raw['HbA1c_Porcentaje'].isna().mean() * 100), 2),
    'nulls_tfg_pct': round(float(df_raw['TFG_mL_min'].isna().mean() * 100), 2),
    'nulls_imc_pct': round(float(df_raw['IMC'].isna().mean() * 100), 2),
    'nulls_adh_pct': round(float(df_raw['Adherencia_Farmacologica'].isna().mean() * 100), 2),
}

# ==============================================================================
# FASE 4 Y FASE 5 ML: LIMPIEZA, CODIFICACIÓN E INGENIERÍA DE CARACTERÍSTICAS
# ==============================================================================
df_fe = df_clean_base.copy()
df_fe['Sexo_Bin'] = (df_fe['Sexo'] == 'Masculino').astype(int)
df_fe['Zona_Rural_Bin'] = (df_fe['Zona_Residencia'] == 'Rural').astype(int)
df_fe['Adherencia_Ord'] = df_fe['Adherencia_Farmacologica'].map({'Baja': 0, 'Media': 1, 'Alta': 2})

# Nuevas características creadas en Fase 5 (Ingeniería de Características)
df_fe['Presion_Pulso_mmHg'] = df_fe['PAS_mmHg'] - df_fe['PAD_mmHg']
df_fe['Carga_Comorbilidad'] = df_fe['Diabetes_Mellitus'] + df_fe['Hipertension_Arterial'] + (df_fe['Estadio_ERC_Base'] >= 3).astype(int)
df_fe['Indice_Riesgo_CardioRenal'] = (df_fe['HbA1c_Porcentaje'] * np.log1p(df_fe['Albuminuria_ACR_mg_g'])) / (df_fe['TFG_mL_min'] / 60.0)
df_fe['Sinergia_Gluco_Renal'] = ((df_fe['HbA1c_Porcentaje'] >= 8.2) & (df_fe['TFG_mL_min'] <= 55.0)).astype(int)
df_fe['Sinergia_HTA_NoAdh'] = ((df_fe['PAS_mmHg'] >= 152.0) & (df_fe['Adherencia_Ord'] == 0)).astype(int)
df_fe['Sinergia_Albuminuria_ERC'] = ((df_fe['Albuminuria_ACR_mg_g'] >= 65.0) & (df_fe['Estadio_ERC_Base'] >= 3)).astype(int)

feature_cols = [
    'Edad', 'Sexo_Bin', 'Zona_Rural_Bin', 'Diabetes_Mellitus', 'Hipertension_Arterial',
    'Estadio_ERC_Base', 'IMC', 'HbA1c_Porcentaje', 'PAS_mmHg', 'PAD_mmHg',
    'TFG_mL_min', 'Creatinina_mg_dL', 'Albuminuria_ACR_mg_g', 'Colesterol_Alto',
    'Tabaquismo', 'Actividad_Fisica', 'Adherencia_Ord', 'Consultas_Control_12m',
    'Urgencias_Previas_12m', 'Polifarmacia_Num_Meds', 'Salud_General_BRFSS',
    'Presion_Pulso_mmHg', 'Carga_Comorbilidad', 'Indice_Riesgo_CardioRenal',
    'Sinergia_Gluco_Renal', 'Sinergia_HTA_NoAdh', 'Sinergia_Albuminuria_ERC'
]

# ==============================================================================
# FASE 6 ML: DIVIDIR LOS DATOS (70% Entrenamiento, 15% Validación, 15% Prueba)
# ==============================================================================
pos_idx = np.where(df_fe['Complicacion_6_12m'].values == 1)[0]
neg_idx = np.where(df_fe['Complicacion_6_12m'].values == 0)[0]
np.random.shuffle(pos_idx)
np.random.shuffle(neg_idx)

def split_indices(indices, p_train=0.70, p_val=0.15):
    n = len(indices)
    n_tr = int(round(n * p_train))
    n_va = int(round(n * p_val))
    return indices[:n_tr], indices[n_tr:n_tr+n_va], indices[n_tr+n_va:]

pos_tr, pos_va, pos_te = split_indices(pos_idx)
neg_tr, neg_va, neg_te = split_indices(neg_idx)

train_idx = np.concatenate([pos_tr, neg_tr])
val_idx = np.concatenate([pos_va, neg_va])
test_idx = np.concatenate([pos_te, neg_te])
np.random.shuffle(train_idx)
np.random.shuffle(val_idx)
np.random.shuffle(test_idx)

X_raw = df_fe[feature_cols].values.astype(float)
y_all = df_fe['Complicacion_6_12m'].values.astype(int)

X_train_raw, y_train = X_raw[train_idx], y_all[train_idx]
X_val_raw, y_val = X_raw[val_idx], y_all[val_idx]
X_test_raw, y_test = X_raw[test_idx], y_all[test_idx]

mean_tr = X_train_raw.mean(axis=0)
std_tr = X_train_raw.std(axis=0) + 1e-8
X_train = (X_train_raw - mean_tr) / std_tr
X_val = (X_val_raw - mean_tr) / std_tr
X_test = (X_test_raw - mean_tr) / std_tr

# ==============================================================================
# FASES 7, 8 y 9 ML: SELECCIÓN DE ALGORITMOS, ENTRENAMIENTO Y AJUSTE
# ==============================================================================

# 1. Modelo Baseline: Regresión Logística Regularizada (solo variables lineales originales sin términos de interacción)
lin_idx = list(range(21))
X_tr_lin = np.column_stack([np.ones(len(X_train)), X_train[:, lin_idx]])
X_va_lin = np.column_stack([np.ones(len(X_val)), X_val[:, lin_idx]])
X_te_lin = np.column_stack([np.ones(len(X_test)), X_test[:, lin_idx]])

w_pos = len(y_train) / (2.0 * np.sum(y_train == 1))
w_neg = len(y_train) / (2.0 * np.sum(y_train == 0))
sample_weights_tr = np.where(y_train == 1, w_pos, w_neg)

def loss_logreg(w, X, y, sw, reg_lambda=10.0):
    p = np.clip(expit(X @ w), 1e-7, 1 - 1e-7)
    ll = -np.mean(sw * (y * np.log(p) + (1 - y) * np.log(1 - p)))
    reg = 0.5 * reg_lambda * np.sum(w[1:] ** 2) / len(y)
    return ll + reg

def grad_logreg(w, X, y, sw, reg_lambda=10.0):
    p = expit(X @ w)
    g = (X.T @ (sw * (p - y))) / len(y)
    g[1:] += (reg_lambda * w[1:]) / len(y)
    return g

res_lr = minimize(loss_logreg, np.zeros(X_tr_lin.shape[1]), args=(X_tr_lin, y_train, sample_weights_tr, 10.0),
                  jac=grad_logreg, method='BFGS')
w_lr = res_lr.x
prob_val_lr = expit(X_va_lin @ w_lr)
prob_test_lr = expit(X_te_lin @ w_lr)

# 2. Árboles de Decisión Vectorizados (CART) para Random Forest y XGBoost
class FastDecisionTree:
    def __init__(self, max_depth=5, min_samples_leaf=20, max_features=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None

    def _fit_node(self, X, y, w, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or np.all(y == y[0]):
            val = np.sum(w * y) / (np.sum(w) + 1e-8)
            return {'leaf': True, 'val': float(val)}

        n_samples, n_features = X.shape
        feat_indices = np.random.choice(n_features, self.max_features, replace=False) if self.max_features else np.arange(n_features)

        best_gain = -1.0
        best_feat = None
        best_thresh = None

        base_mean = np.sum(w * y) / (np.sum(w) + 1e-8)
        base_var = np.sum(w * (y - base_mean) ** 2)

        for f in feat_indices:
            col = X[:, f]
            u_vals = np.unique(col)
            if len(u_vals) <= 6:
                thresh_cands = u_vals[:-1]
            else:
                thresh_cands = np.unique(np.percentile(col, [10, 20, 35, 50, 65, 80, 90]))
            for thresh in thresh_cands:
                left_mask = col <= thresh
                n_left = np.sum(left_mask)
                if n_left < self.min_samples_leaf or (n_samples - n_left) < self.min_samples_leaf:
                    continue
                w_l, y_l = w[left_mask], y[left_mask]
                w_r, y_r = w[~left_mask], y[~left_mask]
                mean_l = np.sum(w_l * y_l) / (np.sum(w_l) + 1e-8)
                mean_r = np.sum(w_r * y_r) / (np.sum(w_r) + 1e-8)
                var_l = np.sum(w_l * (y_l - mean_l) ** 2)
                var_r = np.sum(w_r * (y_r - mean_r) ** 2)
                gain = base_var - (var_l + var_r)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thresh = float(thresh)

        if best_feat is None or best_gain <= 1e-6:
            val = np.sum(w * y) / (np.sum(w) + 1e-8)
            return {'leaf': True, 'val': float(val)}

        left_mask = X[:, best_feat] <= best_thresh
        return {
            'leaf': False,
            'feat': best_feat,
            'thresh': best_thresh,
            'gain': best_gain,
            'left': self._fit_node(X[left_mask], y[left_mask], w[left_mask], depth + 1),
            'right': self._fit_node(X[~left_mask], y[~left_mask], w[~left_mask], depth + 1)
        }

    def fit(self, X, y, w=None):
        if w is None:
            w = np.ones(len(y))
        self.tree = self._fit_node(X, y, w, 0)
        return self

    def _predict_node(self, node, X):
        if node['leaf']:
            return np.full(len(X), node['val'])
        mask = X[:, node['feat']] <= node['thresh']
        out = np.empty(len(X))
        if np.any(mask):
            out[mask] = self._predict_node(node['left'], X[mask])
        if np.any(~mask):
            out[~mask] = self._predict_node(node['right'], X[~mask])
        return out

    def predict(self, X):
        return self._predict_node(self.tree, X)

    def accumulate_importance(self, node, imp_array):
        if not node['leaf']:
            imp_array[node['feat']] += node['gain']
            self.accumulate_importance(node['left'], imp_array)
            self.accumulate_importance(node['right'], imp_array)

# Entrenar Random Forest
print("[2/6] Regresión logística entrenada. Entrenando Random Forest (puede tardar unos minutos)...")
n_trees_rf = 70
rf_trees = []
rf_importances = np.zeros(X_train.shape[1])
for t in range(n_trees_rf):
    boot_idx = np.random.choice(len(X_train), len(X_train), replace=True)
    tree = FastDecisionTree(max_depth=6, min_samples_leaf=22, max_features=11)
    tree.fit(X_train[boot_idx], y_train[boot_idx], sample_weights_tr[boot_idx])
    tree.accumulate_importance(tree.tree, rf_importances)
    rf_trees.append(tree)

prob_val_rf = np.mean([t.predict(X_val) for t in rf_trees], axis=0)
prob_test_rf = np.mean([t.predict(X_test) for t in rf_trees], axis=0)

# 3. Entrenar XGBoost / Gradient Boosting Classifier
print("[3/6] Random Forest listo. Entrenando Gradient Boosting propio (puede tardar unos minutos)...")
n_rounds_xgb = 90
lr_xgb = 0.10
init_p = np.sum(sample_weights_tr * y_train) / np.sum(sample_weights_tr)
init_logit = np.log(init_p / (1.0 - init_p))

f_train = np.full(len(X_train), init_logit)
f_val = np.full(len(X_val), init_logit)
f_test = np.full(len(X_test), init_logit)

xgb_trees = []
xgb_importances = np.zeros(X_train.shape[1])

for m in range(n_rounds_xgb):
    p_curr = expit(f_train)
    residuals = (y_train - p_curr)
    hessian_w = sample_weights_tr * (p_curr * (1.0 - p_curr) + 0.04)
    target_step = residuals / (p_curr * (1.0 - p_curr) + 0.04)
    sub_idx = np.random.choice(len(X_train), int(0.85 * len(X_train)), replace=False)
    tree = FastDecisionTree(max_depth=5, min_samples_leaf=18, max_features=16)
    tree.fit(X_train[sub_idx], target_step[sub_idx], hessian_w[sub_idx])
    tree.accumulate_importance(tree.tree, xgb_importances)
    f_train += lr_xgb * tree.predict(X_train)
    f_val += lr_xgb * tree.predict(X_val)
    f_test += lr_xgb * tree.predict(X_test)
    xgb_trees.append(tree)

prob_val_xgb = expit(f_val)
prob_test_xgb = expit(f_test)

# ==============================================================================
# FASE 10 ML: EVALUACIÓN DE LOS MODELOS Y MATRICES DE CONFUSIÓN
# ==============================================================================
def compute_auc(y_true, y_prob):
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    comp = (pos_scores[:, None] > neg_scores[None, :]).astype(float) + 0.5 * (pos_scores[:, None] == neg_scores[None, :]).astype(float)
    return float(np.sum(comp) / (len(pos_scores) * len(neg_scores)))

def get_roc_curve(y_true, y_prob):
    thresholds = np.linspace(1.0, 0.0, 200)
    tpr_list, fpr_list = [], []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    return np.array(fpr_list), np.array(tpr_list)

def evaluate_model(y_true, y_prob, thresh=0.50):
    pred = (y_prob >= thresh).astype(int)
    vp = int(np.sum((pred == 1) & (y_true == 1)))
    vn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    total = len(y_true)
    acc = (vp + vn) / total
    prec = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    rec = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    auc = compute_auc(y_true, y_prob)
    return {
        'VP': vp, 'VN': vn, 'FP': fp, 'FN': fn, 'Total': total,
        'Accuracy': round(acc, 4), 'Precision': round(prec, 4),
        'Recall': round(rec, 4), 'F1': round(f1, 4), 'ROC_AUC': round(auc, 4)
    }

# Búsqueda de umbral óptimo en validación para equilibrar Recall >= 75% y F1
def find_best_thresh(y_va, p_va):
    best_th = 0.50
    best_f1 = -1.0
    for th in np.linspace(0.35, 0.75, 41):
        m = evaluate_model(y_va, p_va, thresh=th)
        if m['Recall'] >= 0.72 and m['F1'] > best_f1:
            best_f1 = m['F1']
            best_th = th
    return best_th

th_lr = find_best_thresh(y_val, prob_val_lr)
th_rf = find_best_thresh(y_val, prob_val_rf)
th_xgb = find_best_thresh(y_val, prob_val_xgb)

metrics_lr_test = evaluate_model(y_test, prob_test_lr, thresh=th_lr)
metrics_rf_test = evaluate_model(y_test, prob_test_rf, thresh=th_rf)
metrics_xgb_test = evaluate_model(y_test, prob_test_xgb, thresh=th_xgb)

# Conservar la relación entre cada fila de prueba y las predicciones de los modelos.
reporte = df_clean_base.iloc[test_idx].copy().reset_index(drop=True)
for nombre, puntajes, umbral in [
    ('Regresion_Logistica', prob_test_lr, th_lr),
    ('Random_Forest', prob_test_rf, th_rf),
    ('Gradient_Boosting', prob_test_xgb, th_xgb),
]:
    pred = (puntajes >= umbral).astype(int)
    reporte[f'Puntaje_{nombre}'] = puntajes
    reporte[f'Umbral_{nombre}'] = umbral
    reporte[f'Prediccion_{nombre}'] = pred
    reporte[f'Resultado_{nombre}'] = np.select(
        [(y_test == 1) & (pred == 1), (y_test == 1) & (pred == 0),
         (y_test == 0) & (pred == 1)],
        ['VP - Identificado', 'FN - No identificado', 'FP - Alerta incorrecta'],
        default='VN - Sin complicacion identificada')
reporte.to_csv(os.path.join(OUTPUT_DIR, 'predicciones_pacientes_prueba.csv'),
               index=False, encoding='utf-8-sig')
print(f"Reporte individual guardado: {len(reporte)} pacientes de prueba.")

metrics_lr_val = evaluate_model(y_val, prob_val_lr, thresh=th_lr)
metrics_rf_val = evaluate_model(y_val, prob_val_rf, thresh=th_rf)
metrics_xgb_val = evaluate_model(y_val, prob_val_xgb, thresh=th_xgb)

naive_acc = round(float(np.mean(y_test == 0)), 4)

xgb_imp_norm = xgb_importances / (np.sum(xgb_importances) + 1e-8)
imp_df = pd.DataFrame({'Variable': feature_cols, 'Importancia': xgb_imp_norm}).sort_values('Importancia', ascending=False)

results_summary = {
    'eda': eda_stats,
    'splits': {
        'train_total': int(len(y_train)), 'train_pos': int(np.sum(y_train == 1)), 'train_neg': int(np.sum(y_train == 0)),
        'val_total': int(len(y_val)), 'val_pos': int(np.sum(y_val == 1)), 'val_neg': int(np.sum(y_val == 0)),
        'test_total': int(len(y_test)), 'test_pos': int(np.sum(y_test == 1)), 'test_neg': int(np.sum(y_test == 0)),
    },
    'thresholds': {'LR': round(float(th_lr), 2), 'RF': round(float(th_rf), 2), 'XGB': round(float(th_xgb), 2)},
    'test_metrics': {
        'Logistic_Regression': metrics_lr_test,
        'Random_Forest': metrics_rf_test,
        'Gradient_Boosting_Propio': metrics_xgb_test,
        'Naive_Baseline': {'Accuracy': naive_acc, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'ROC_AUC': 0.50}
    },
    'val_metrics': {
        'Logistic_Regression': metrics_lr_val,
        'Random_Forest': metrics_rf_val,
        'Gradient_Boosting_Propio': metrics_xgb_val
    },
    'top_features': imp_df.head(10).to_dict(orient='records')
}

with open(os.path.join(INTERMEDIATE_DIR, "ml_results.json"), "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)
print("[4/6] Métricas guardadas en ml_results.json")
for _name, _m in results_summary['test_metrics'].items():
    print(f"      {_name:20s} Acc={_m['Accuracy']}  Recall={_m['Recall']}  F1={_m['F1']}  AUC={_m['ROC_AUC']}")
print("[5/6] Generando figuras...")

# ==============================================================================
# GENERACIÓN DE GRÁFICOS DE ALTA RESOLUCIÓN (180 DPI) PARA LOS ENTREGABLES
# ==============================================================================

# FIGURA 1: EDA - Desbalance de Clases y Distribución de Biomarcadores (Fase 3 ML)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=180)
counts = df_clean_base['Complicacion_6_12m'].value_counts().sort_index()
bars = axes[0].bar(['Sin Complicación\n(Clase 0)', 'Con Complicación\n6-12m (Clase 1)'], counts.values,
                   color=['#0F3D5E', '#D97706'], width=0.55, edgecolor='none')
axes[0].set_title(f'A. Desbalance de Variable Objetivo\n(N = {len(df_clean_base):,} pacientes)', fontsize=10.5, fontweight='bold', color='#0F3D5E')
axes[0].set_ylabel('Número de Pacientes', fontsize=9.5)
axes[0].set_ylim(0, max(counts.values) * 1.2)
for bar in bars:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, h + max(counts.values) * 0.02, f"{h:,}\n({100*h/len(df_clean_base):.1f}%)",
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1E293B')
axes[0].grid(axis='y', linestyle='--', alpha=0.4)

sns.kdeplot(data=df_clean_base[df_clean_base['Complicacion_6_12m']==0], x='HbA1c_Porcentaje',
            fill=True, color='#0F3D5E', alpha=0.35, label='Sin Complicación (0)', ax=axes[1], linewidth=1.8)
sns.kdeplot(data=df_clean_base[df_clean_base['Complicacion_6_12m']==1], x='HbA1c_Porcentaje',
            fill=True, color='#D97706', alpha=0.45, label='Con Complicación (1)', ax=axes[1], linewidth=1.8)
axes[1].axvline(8.2, color='#DC2626', linestyle='--', linewidth=1.2, label='Umbral Riesgo (8.2%)')
axes[1].set_title('B. Densidad de HbA1c (%) por Desenlace', fontsize=10.5, fontweight='bold', color='#0F3D5E')
axes[1].set_xlabel('Hemoglobina Glicosilada HbA1c (%)', fontsize=9.5)
axes[1].set_ylabel('Densidad', fontsize=9.5)
axes[1].legend(fontsize=8, loc='upper right')
axes[1].grid(linestyle='--', alpha=0.4)

sns.boxplot(data=df_clean_base, x='Estadio_ERC_Base', y='TFG_mL_min', hue='Complicacion_6_12m',
            palette={0: '#0F3D5E', 1: '#D97706'}, ax=axes[2], fliersize=2)
axes[2].set_title('C. Función Renal (TFG) por Estadio ERC', fontsize=10.5, fontweight='bold', color='#0F3D5E')
axes[2].set_xlabel('Estadio ERC Basal', fontsize=9.5)
axes[2].set_ylabel('TFG (mL/min/1.73m²)', fontsize=9.5)
handles, _ = axes[2].get_legend_handles_labels()
axes[2].legend(handles, ['No Complicación (0)', 'Complicación (1)'], fontsize=8, loc='upper right')
axes[2].grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(INTERMEDIATE_DIR, "fig1_eda_distribuciones.png"), bbox_inches='tight')
plt.close()

# FIGURA 2: Auditoría de Calidad de Datos (Nulos y Correlaciones - Fase 3 y Fase 4 ML)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), dpi=180)
null_vars = ['HbA1c_Porcentaje', 'TFG_mL_min', 'IMC', 'Adherencia_Farmacologica', 'PAS_mmHg', 'Creatinina_mg_dL']
null_pcts = [df_raw[c].isna().mean() * 100 for c in null_vars]
labels_null = ['HbA1c (%)', 'TFG (mL/min)', 'IMC (kg/m²)', 'Adherencia Farm.', 'PAS (mmHg)', 'Creatinina (mg/dL)']
y_pos = np.arange(len(labels_null))
bars_n = axes[0].barh(y_pos, null_pcts, color=['#008080' if v > 0 else '#94A3B8' for v in null_pcts], height=0.55)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(labels_null, fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('Porcentaje de Valores Faltantes (%)', fontsize=9.5)
axes[0].set_title('A. Valores Faltantes en el Dataset de Entrada', fontsize=10.5, fontweight='bold', color='#0F3D5E')
axes[0].set_xlim(0, 8.0)
for bar in bars_n:
    w = bar.get_height() if False else bar.get_width()
    axes[0].text(w + 0.15, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
axes[0].grid(axis='x', linestyle='--', alpha=0.4)

corr_cols = ['Edad', 'IMC', 'HbA1c_Porcentaje', 'PAS_mmHg', 'TFG_mL_min', 'Albuminuria_ACR_mg_g', 'Urgencias_Previas_12m', 'Complicacion_6_12m']
corr_names = ['Edad', 'IMC', 'HbA1c', 'PAS', 'TFG', 'Albuminuria', 'Urgencias', 'Complicación']
corr_mat = df_clean_base[corr_cols].corr()
corr_mat.index = corr_names
corr_mat.columns = corr_names
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=axes[1], annot_kws={"size": 7.5})
axes[1].set_title('B. Matriz de Correlación de Pearson (Variables Clínicas)', fontsize=10.5, fontweight='bold', color='#0F3D5E')
axes[1].tick_params(axis='x', rotation=30, labelsize=8.5)
axes[1].tick_params(axis='y', rotation=0, labelsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(INTERMEDIATE_DIR, "fig2_calidad_y_correlacion.png"), bbox_inches='tight')
plt.close()

# FIGURA 3: Diagrama de Gantt del Cronograma de Trabajo (Semanas 1 a 12 - Entregable 3 Fase 1)
fig, ax = plt.subplots(figsize=(12.5, 4.8), dpi=180)
tasks = [
    ("F12 ML: Despliegue App Streamlit/API & Pitch Final", 10, 3, "#475569"),
    ("F11 ML: Validación de Generalización & Auditoría Ética", 8, 3, "#475569"),
    ("F10 ML: Evaluación (Matriz Confusión, ROC-AUC, Recall)", 6, 2, "#008080"),
    ("F08-F09 ML: Entrenamiento y Ajuste Hiperparámetros", 5, 2, "#008080"),
    ("F06-F07 ML: Partición 70/15/15 y Selección Algoritmos", 4, 2, "#008080"),
    ("F04-F05 ML: Limpieza, Imputación e Ing. Características", 3, 2, "#008080"),
    ("F03 ML: Análisis Exploratorio de Datos (EDA en Notebook)", 1.5, 1.5, "#0F3D5E"),
    ("F02 ML: Recolección e Integración BRFSS + RIPS + HCE", 1, 1.5, "#0F3D5E"),
    ("F01 ML: Definición del Problema, Estado del Arte y Propuesta", 1, 1, "#0F3D5E"),
]
for idx, (label, start, duration, color) in enumerate(tasks):
    ax.barh(idx, duration, left=start - 0.5, height=0.58, color=color, alpha=0.92, edgecolor='white')
    ax.text(start - 0.5 + duration/2, idx, f"Sem {int(np.ceil(start))}–{int(start+duration-1)}",
            ha='center', va='center', color='white', fontsize=8, fontweight='bold')

ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t[0] for t in tasks], fontsize=8.8, fontweight='bold', color='#1E293B')
ax.set_xticks(range(1, 13))
ax.set_xticklabels([f"Sem {i}" for i in range(1, 13)], fontsize=8.8)
ax.set_xlim(0.5, 12.5)
ax.axvline(2.5, color='#D97706', linestyle='--', linewidth=1.5)
ax.axvline(7.5, color='#DC2626', linestyle='--', linewidth=1.5)
ax.text(1.5, len(tasks)-0.3, "ENTREGA FASE 1\n(Sem 1–2)", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#0F3D5E')
ax.text(5.0, len(tasks)-0.3, "ENTREGA FASE 2 (ACTUAL)\n(Sem 3–7 | Hasta Fase 10 ML)", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#008080')
ax.text(10.0, len(tasks)-0.3, "ENTREGA FASE 3\n(Sem 8–12 | Fases 11–12 ML)", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#475569')
ax.set_title("Cronograma General del Proyecto de Aula — Grupo 01 (Alineado con las 12 Fases de ML)", fontsize=11, fontweight='bold', color='#0F3D5E', pad=25)
ax.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(INTERMEDIATE_DIR, "fig3_cronograma_gantt.png"), bbox_inches='tight')
plt.close()

# FIGURA 4: Curvas ROC, Importancia de Características y Matrices de Confusión (Fases 5, 9 y 10 ML)
fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), dpi=180)

fpr_lr, tpr_lr = get_roc_curve(y_test, prob_test_lr)
fpr_rf, tpr_rf = get_roc_curve(y_test, prob_test_rf)
fpr_xgb, tpr_xgb = get_roc_curve(y_test, prob_test_xgb)

axes[0].plot(fpr_xgb, tpr_xgb, color='#008080', linewidth=2.4, label=f"Gradient Boosting propio (AUC = {metrics_xgb_test['ROC_AUC']:.3f})")
axes[0].plot(fpr_rf, tpr_rf, color='#0F3D5E', linewidth=2.0, label=f"Random Forest (AUC = {metrics_rf_test['ROC_AUC']:.3f})")
axes[0].plot(fpr_lr, tpr_lr, color='#D97706', linewidth=1.8, linestyle='-.', label=f"Regresión Logística (AUC = {metrics_lr_test['ROC_AUC']:.3f})")
axes[0].plot([0, 1], [0, 1], color='#94A3B8', linestyle='--', linewidth=1.2, label="Azar / Modelo Ingenuo (AUC = 0.500)")
axes[0].set_title(f"A. Curvas ROC en Conjunto de Prueba\n(N = {len(y_test):,} pacientes)", fontsize=10.5, fontweight='bold', color='#0F3D5E')
axes[0].set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=9)
axes[0].set_ylabel("Tasa de Verdaderos Positivos (Recall / Sensibilidad)", fontsize=9)
axes[0].legend(loc="lower right", fontsize=8)
axes[0].grid(linestyle='--', alpha=0.4)

cm = np.array([[metrics_xgb_test['VP'], metrics_xgb_test['FN']],
               [metrics_xgb_test['FP'], metrics_xgb_test['VN']]])
annot_cm = np.array([
    [f"VP = {metrics_xgb_test['VP']}\n(Verdadero Positivo)", f"FN = {metrics_xgb_test['FN']}\n(Falso Negativo)"],
    [f"FP = {metrics_xgb_test['FP']}\n(Falso Positivo)", f"VN = {metrics_xgb_test['VN']}\n(Verdadero Negativo)"]
])
sns.heatmap(cm, annot=annot_cm, fmt="", cmap="Blues", cbar=False, ax=axes[1],
            xticklabels=['Predicho: Sí Complica', 'Predicho: No Complica'],
            yticklabels=['Real: Sí Complica', 'Real: No Complica'],
            annot_kws={"size": 9.5, "weight": "bold"})
axes[1].set_title(f"B. Matriz de Confusión Gradient Boosting\nEvaluación sobre {len(y_test):,} pacientes de prueba", fontsize=10.5, fontweight='bold', color='#0F3D5E')

top8 = imp_df.head(8).iloc[::-1]
label_map = {
    'Indice_Riesgo_CardioRenal': 'Índice Riesgo Cardio-Renal (FE)',
    'Sinergia_Gluco_Renal': 'Sinergia HbA1c≥8.2 & TFG≤55 (FE)',
    'Sinergia_Albuminuria_ERC': 'Sinergia Albuminuria & ERC≥3 (FE)',
    'Sinergia_HTA_NoAdh': 'Sinergia PAS≥152 & No Adh. (FE)',
    'TFG_mL_min': 'TFG Basal (mL/min/1.73m²)',
    'HbA1c_Porcentaje': 'Hemoglobina Glicosilada HbA1c (%)',
    'Albuminuria_ACR_mg_g': 'Albuminuria ACR (mg/g)',
    'Adherencia_Ord': 'Adherencia Farmacológica',
    'Urgencias_Previas_12m': 'Urgencias Previas (12m)',
    'PAS_mmHg': 'Presión Arterial Sistólica (mmHg)',
    'Carga_Comorbilidad': 'Carga de Comorbilidad (FE)'
}
clean_labels = [label_map.get(v, v) for v in top8['Variable']]
axes[2].barh(clean_labels, top8['Importancia'] * 100, color='#0F3D5E', height=0.6)
axes[2].set_title("C. Importancia Relativa de Variables (%)\nGradient Boosting propio (Ganancia de Información)", fontsize=10.5, fontweight='bold', color='#0F3D5E')
axes[2].set_xlabel("Importancia Relativa (%)", fontsize=9)
axes[2].grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(INTERMEDIATE_DIR, "fig4_evaluacion_modelos_fase10.png"), bbox_inches='tight')
plt.close()
print(f"[6/6] ¡Listo! Revisa la carpeta: {os.path.join(BASE_DIR, 'resultados')}")