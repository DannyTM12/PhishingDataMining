#Librerias, pandas para manejo de datos
import pandas as pd

#Importamos los datasets originales
naz = pd.read_csv("Nazario.csv")
nig = pd.read_csv("Nigerian_Fraud.csv")
spa = pd.read_csv("SpamAssasin.csv")

#Agregamos una columna de origen, para determinar de qué dataset viene la info
naz['origen'] = 'Nazario'
nig['origen'] = 'Nigerian_Fraud'
spa['origen'] = 'SpamAssasin'

#Unimos los datasets en uno solo
data = pd.concat([naz, nig, spa], ignore_index=True)

#verificamos el balance de clases
print("Distribución de clases en el dataset combinado: ")
print(data['label'].value_counts())

#Verificar duplicados y valores faltantes
print("Duplicados por body:", data.duplicated(subset=["body"]).sum())
print("Valores faltantes:")
print(data.isna().mean().round(3))

#Guardamos el dataset combinado en un nuevo archivo CSV
data.to_csv("Combined_Dataset.csv", index=False)