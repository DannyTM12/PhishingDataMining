import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("Cargando dataset limpio...")
data = pd.read_csv("Cleaned_Featured_Dataset.csv")
print("Filas:", len(data))

# ===============================
# 1. DEFINIR FEATURES Y TARGET
# ===============================

target = "label"

# Columnas numéricas que creaste en el script de limpieza/FE
feature_cols = [
    "subject_len",
    "body_len",
    "subject_words",
    "body_words",
    "subject_upper",
    "body_upper",
    "digits_count",
    "special_chars",
    "urgency_score",
    "email_count",
    "url_count",
    "urls"            # la original, por si aporta algo
]

X = data[feature_cols]
y = data[target]

print("\nShapes:")
print("X:", X.shape)
print("y:", y.shape)

# ===============================
# 2. TRAIN / TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/Test sizes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

# ===============================
# 3. ESCALADO PARA REGRESIÓN LOGÍSTICA
# ===============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 4. MODELO 1: REGRESIÓN LOGÍSTICA
# ===============================

print("\n=== MODELO 1: Regresión Logística ===")

log_clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)
log_clf.fit(X_train_scaled, y_train)
y_pred_log = log_clf.predict(X_test_scaled)
y_proba_log = log_clf.predict_proba(X_test_scaled)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall   :", recall_score(y_test, y_pred_log))
print("F1       :", f1_score(y_test, y_pred_log))
print("AUC      :", roc_auc_score(y_test, y_proba_log))

print("\nClassification report (Logistic):")
print(classification_report(y_test, y_pred_log))

# ===============================
# 5. MODELO 2: RANDOM FOREST
# ===============================

print("\n=== MODELO 2: RandomForest ===")

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)  # RandomForest no necesita escalado
y_pred_rf = rf_clf.predict(X_test)
y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1       :", f1_score(y_test, y_pred_rf))
print("AUC      :", roc_auc_score(y_test, y_proba_rf))

print("\nClassification report (RandomForest):")
print(classification_report(y_test, y_pred_rf))

print("\nModelado baseline completo.")