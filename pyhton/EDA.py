import pandas as pd

# Cargar el dataset combinado
data = pd.read_csv("Combined_Dataset.csv")

print("\n=== 1. Vista general del dataset ===")
print(data.info())

print("\n=== 2. Primeras 5 filas ===")
print(data.head())

print("\n=== 3. Distribución de clases ===")
print(data['label'].value_counts())

print("\n=== 4. Valores nulos por columna ===")
print(data.isna().sum())

print("\n=== 5. Porcentaje de valores nulos ===")
print((data.isna().mean()*100).round(2))

print("\n=== 6. Duplicados ===")
print("Duplicados por cuerpo:", data.duplicated(subset=['body']).sum())
print("Duplicados completos:", data.duplicated().sum())

print("\n=== 7. Longitud de subject y body ===")
data['subject_len'] = data['subject'].astype(str).apply(len)
data['body_len'] = data['body'].astype(str).apply(len)
print(data[['subject_len', 'body_len']].describe())

print("\n=== 8. Conteo de URLs ===")
data['num_urls'] = data['urls'].astype(str).apply(lambda x: len(str(x).split()) if x != 'nan' else 0)
print(data['num_urls'].describe())

# Guardamos con nuevas columnas
data.to_csv("Combined_Dataset_Features.csv", index=False)
print("\n>>> Archivo guardado: Combined_Dataset_Features.csv")
