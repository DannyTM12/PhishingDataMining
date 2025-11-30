import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap

print("Cargando dataset limpio con features...")
data = pd.read_csv("Cleaned_Featured_Dataset.csv")
print("Filas:", len(data))

# ====================================================
# 1. SELECCIONAR SOLO FEATURES NUMÉRICAS
# ====================================================

numeric_features = [
    "subject_len", "body_len",
    "subject_words", "body_words",
    "subject_upper", "body_upper",
    "digits_count", "special_chars",
    "urgency_score", "email_count", "url_count",
    "urls"   # <- tu columna original, también es numérica
]

X = data[numeric_features]
y = data["label"]

print("\nShapes:")
print("X:", X.shape)
print("y:", y.shape)

# ====================================================
# 2. TRAIN / TEST SPLIT
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ====================================================
# 3. RANDOM FOREST PARA INTERPRETABILIDAD
# ====================================================

print("\nEntrenando RandomForest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# ====================================================
# 4. IMPORTANCIA DE FEATURES
# ====================================================

importances = pd.DataFrame({
    "feature": numeric_features,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nIMPORTANCIA DE VARIABLES:")
print(importances)

importances.to_csv("Feature_Importance_RF.csv", index=False)

# ====================================================
# 5. INTERPRETABILIDAD SHAP
# ====================================================

print("\nCalculando valores SHAP (esto puede tardar un poco)...")

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Compatibilidad: SHAP a veces regresa lista (binario) o solo un array 2D
if isinstance(shap_values, list):
    # Binario: tomamos la clase 1 (phishing)
    shap_to_plot = shap_values[1]
else:
    # Ya es matriz (n_samples, n_features)
    shap_to_plot = shap_values

# Guardar un SHAP summary plot
print("Generando SHAP summary plot...")
shap.summary_plot(shap_values[1], X_test, show=False)

print("\nInterpretabilidad COMPLETA.")
print(" - Archivo guardado: Feature_Importance_RF.csv")