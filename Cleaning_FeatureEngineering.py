import pandas as pd
import numpy as np
import re

# ===============================
#   1. CARGAR DATASET INTEGRADO
# ===============================
print("Cargando dataset...")
data = pd.read_csv("Combined_Dataset_Features.csv")
print("Filas cargadas:", len(data))

# ===============================
#   2. LIMPIEZA DE DATOS
# ===============================

# --- Quitar espacios y estandarizar texto ---
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)        # espacios múltiples
    text = text.replace("\n", " ")          # saltos de línea
    return text.strip()

print("Limpiando columnas de texto...")
data["sender"]   = data["sender"].apply(clean_text)
data["receiver"] = data["receiver"].apply(clean_text)
data["subject"]  = data["subject"].apply(clean_text)
data["body"]     = data["body"].apply(clean_text)


# --- Manejo de valores faltantes ---
print("Rellenando valores faltantes...")
cols_to_fill = {
    "sender": "unknown_sender",
    "receiver": "unknown_receiver",
    "date": "unknown_date",
    "subject": "no_subject",
    "body": ""
}
data = data.fillna(cols_to_fill)


# ===============================
#   3. FEATURE ENGINEERING
# ===============================

print("Generando nuevas características...")

# Longitud de textos
data["subject_len"] = data["subject"].apply(len)
data["body_len"] = data["body"].apply(len)

# Conteo de palabras
data["subject_words"] = data["subject"].apply(lambda x: len(x.split()))
data["body_words"] = data["body"].apply(lambda x: len(x.split()))

# Porcentaje de mayúsculas
def percent_upper(text):
    if len(text) == 0:
        return 0
    return sum(1 for c in text if c.isupper()) / len(text)

data["subject_upper"] = data["subject"].apply(percent_upper)
data["body_upper"] = data["body"].apply(percent_upper)

# Conteo de números
data["digits_count"] = data["body"].apply(lambda x: sum(1 for c in x if c.isdigit()))

# Conteo de caracteres especiales
data["special_chars"] = data["body"].apply(lambda x: len(re.findall(r"[^A-Za-z0-9 ]", x)))

# Detección simple de urgencia (palabras comunes en phishing)
UrgencyWords = ["urgent", "verify", "update", "warning", "suspend", "password", "bank", "security"]
data["urgency_score"] = data["body"].apply(
    lambda text: sum(1 for w in UrgencyWords if w.lower() in text.lower())
)

# Detección de emails en el body
data["email_count"] = data["body"].apply(lambda x: len(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", x)))

# Detección de URL real
data["url_count"] = data["body"].apply(lambda x: len(re.findall(r"https?://\S+", x)))


# ===============================
#   4. FORMATO FINAL
# ===============================

print("Guardando dataset final para modelado...")

data.to_csv("Cleaned_Featured_Dataset.csv", index=False)

print("\n====================================")
print("  Limpieza + Feature Engineering COMPLETO")
print("  Archivo generado: Cleaned_Featured_Dataset.csv")
print("  Filas finales:", len(data))
print("====================================")