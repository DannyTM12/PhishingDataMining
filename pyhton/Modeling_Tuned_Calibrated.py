import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42

print("Cargando dataset limpio con features...")
data = pd.read_csv("Cleaned_Featured_Dataset.csv")
print("Filas:", len(data))

# ===========================
# 1. Definir X, y
# ===========================
# Tomamos solo columnas numéricas y quitamos 'label'
num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove("label")

X = data[num_cols]
y = data["label"]

print("\nShapes:")
print("X:", X.shape)
print("y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain/Test:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

# Función auxiliar para imprimir métricas
def print_metrics(nombre, y_true, y_pred, y_proba):
    print(f"\n=== {nombre} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1       :", f1_score(y_true, y_pred))
    print("AUC      :", roc_auc_score(y_true, y_proba))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred))


# ===========================
# 2. Modelo 1: Logistic Regression (tuneada)
# ===========================
log_clf = LogisticRegression(max_iter=2000)

param_grid_log = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs", "liblinear"]
}

grid_log = GridSearchCV(
    log_clf,
    param_grid_log,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

print("\n>>> Entrenando Logistic Regression (GridSearchCV)...")
grid_log.fit(X_train, y_train)
print("Mejores parámetros LogReg:", grid_log.best_params_)
print("Mejor AUC CV LogReg:", grid_log.best_score_)

best_log = grid_log.best_estimator_
y_pred_log = best_log.predict(X_test)
y_proba_log = best_log.predict_proba(X_test)[:, 1]

print_metrics("LOGISTIC REGRESSION TUNED", y_test, y_pred_log, y_proba_log)


# ===========================
# 3. Modelo 2: RandomForest (tuneado)
# ===========================
rf_clf = RandomForestClassifier(random_state=RANDOM_STATE)

param_grid_rf = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_rf = GridSearchCV(
    rf_clf,
    param_grid_rf,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

print("\n>>> Entrenando RandomForest (GridSearchCV)...")
grid_rf.fit(X_train, y_train)
print("Mejores parámetros RF:", grid_rf.best_params_)
print("Mejor AUC CV RF:", grid_rf.best_score_)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

print_metrics("RANDOM FOREST TUNED", y_test, y_pred_rf, y_proba_rf)


# ===========================
# 4. Modelo 3: Gradient Boosting (tercer método)
# ===========================
gb_clf = GradientBoostingClassifier(random_state=RANDOM_STATE)

param_grid_gb = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5]
}

grid_gb = GridSearchCV(
    gb_clf,
    param_grid_gb,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

print("\n>>> Entrenando GradientBoosting (GridSearchCV)...")
grid_gb.fit(X_train, y_train)
print("Mejores parámetros GB:", grid_gb.best_params_)
print("Mejor AUC CV GB:", grid_gb.best_score_)

best_gb = grid_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test)
y_proba_gb = best_gb.predict_proba(X_test)[:, 1]

print_metrics("GRADIENT BOOSTING TUNED", y_test, y_pred_gb, y_proba_gb)

# ===========================
# 5. Calibración (Platt vs Isotónica) sobre el mejor modelo
#    Aquí tomamos el RANDOM FOREST como candidato.
# ===========================
print("\n>>> Calibrando RandomForest (Platt / Isotonic)...")

cal_platt = CalibratedClassifierCV(best_rf, method="sigmoid", cv=5)
cal_platt.fit(X_train, y_train)

cal_iso = CalibratedClassifierCV(best_rf, method="isotonic", cv=5)
cal_iso.fit(X_train, y_train)

y_proba_platt = cal_platt.predict_proba(X_test)[:, 1]
y_proba_iso   = cal_iso.predict_proba(X_test)[:, 1]

brier_platt = brier_score_loss(y_test, y_proba_platt)
brier_iso   = brier_score_loss(y_test, y_proba_iso)

print("\nBrier score RF sin calibrar :", brier_score_loss(y_test, y_proba_rf))
print("Brier score RF Platt (sigmoid):", brier_platt)
print("Brier score RF Isotonic       :", brier_iso)
print("AUC RF Platt :", roc_auc_score(y_test, y_proba_platt))
print("AUC RF Iso   :", roc_auc_score(y_test, y_proba_iso))

print("\n Modelado tuneado + calibración COMPLETO.")
