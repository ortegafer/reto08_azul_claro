# Reto 8: Optimización de Compresores

Este proyecto está enfocado en la optimización de compresores industriales mediante el análisis de datos y la aplicación de algoritmos genéticos y técnicas de machine learning.

## Estructura del proyecto
- **Datos/Originales/**: Archivos fuente (CSV, logs) de sensores y redes de compresores.
- **Datos/Transformados/**: Archivos procesados y listos para análisis y modelado.
- **Scripts/**: Notebooks para preprocesamiento, análisis, optimización y visualización de resultados.
- **requirements.txt**: Dependencias necesarias para ejecutar los notebooks.

## Instalación
1. Clona este repositorio o descarga el proyecto.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Objetivos principales
- Analizar el funcionamiento de compresores a partir de datos históricos y de sensores.
- Detectar patrones, anomalías y oportunidades de mejora en el consumo energético y la producción de aire.
- Aplicar algoritmos genéticos (pygad) para optimizar la configuración de frecuencias y parámetros de operación.
- Visualizar resultados y comparar escenarios de optimización.

## Uso
- Ejecuta los notebooks en la carpeta `Scripts/` para:
  - Preprocesar y limpiar los datos de sensores y logs.
  - Analizar el comportamiento de los compresores.
  - Aplicar modelos de optimización y machine learning.
  - Visualizar resultados y recomendaciones.

## Notebooks principales
- `1. Preprocesamiento.ipynb`: Limpieza y transformación de datos de sensores y logs.
- `2. Analisis_descriptivo.ipynb`: Exploración y visualización de datos (estadísticas, tendencias, correlaciones).
- `3. Deteccion_de_anomalias.ipynb`: Identificación de comportamientos anómalos en los datos.
- `4-5-6-7. DM_compresor_A/B/C/D.ipynb`: Análisis y modelado específico para cada compresor.
- `8. DM_optimizacion.ipynb`: Algoritmos de optimización (genéticos y otros) aplicados a compresores.

- `9. MYsql_MongoDB_Consultas.ipynb`: Ejemplo de integración de datos y consultas avanzadas (opcional).

- La aplicación Dash se encuentra en el proyecto de Drive ya que necesitaba un csv sacado de las consultas de Data Science. Para la profesora de visu se ha habilitado otro github compratida con ella. 
