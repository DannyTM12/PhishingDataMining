Proyecto Final de Minería de Datos
Detección de Phishing mediante Integración de Múltiples Datasets, Minería de Datos y Visualización Interactiva

Equipo:
César Eduardo Elías del Hoyo
Diego Emanuel Saucedo Ortega
Carlos Daniel Torres Macías

Materia: Minería de Datos
Carrera: Ingeniería en Computación Inteligente
Semestre: 9º
Periodo: Enero–Junio 2025
Profesor: Migue Angel Meza
Proyecto: Analítica de Datos (Tema libre)
Herramientas utilizadas: Excel, Power BI, Looker Studio, Python, Orange Data Mining

Descripción del Proyecto

Este proyecto desarrolla un sistema completo de detección de correos phishing vs legítimos, integrando tres datasets heterogéneos:

Nazario Malware Corpus
Nigerian Fraud Mails
SpamAssassin Email Corpus

Cada dataset poseía estructuras diferentes, por lo que se realizó un proceso completo de:

Limpieza
Unificación de columnas
Generación de nuevas características
Integración final en un dataset consolidado

Después de integrar y transformar los datos, se entrenaron múltiples modelos de clasificación usando Python y Orange Data Mining, se interpretaron los modelos con SHAP, y se generó un dashboard interactivo en Looker Studio y Power BI.


Procesamiento de Datos
Unificación de columnas

Se homologaron las columnas principales:

sender
receiver
subject
body
date
label (0 = legítimo, 1 = phishing)
origen (procedencia del dataset)

Ejecutar cada etapa del pipeline:

python Joining.py
python Cleaning_FeatureEngineering.py
python EDA.py
python Modeling_Baseline.py
python Modeling_Tuned_Calibrated.py
python Modeling_Interpretability.py

Roles del Equipo
Integrante	Rol
César Eduardo Elías del Hoyo	Integración de datasets, ETL, dashboards
Diego Emanuel Saucedo Ortega	Análisis en Excel, Modelado en Orange
Carlos Daniel Torres Macías	Pipeline en Python, interpretabilidad, documentación

Licencia
Proyecto académico para el curso de Minería de Datos.
No tiene restricciones excepto uso ético y no comercial.

Contacto

Para dudas o reproducibilidad:

Carlos Daniel Torres Macías: carlosdtm8@gmail.com

Disponible para seguimiento académico y técnico