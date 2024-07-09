
# CNN para la Detección de Tumores en Imágenes Histopatológicas

Este repositorio contiene los códigos y scripts necesarios para entrenar, evaluar y analizar una Red Neuronal Convolucional (CNN) destinada a la detección temprana de tumores en imágenes histopatológicas de colon y pulmón.

## Descripción

La detección temprana de tumores es crucial para mejorar los resultados del tratamiento y reducir la mortalidad. Este proyecto implementa una CNN para clasificar imágenes histopatológicas de tumores, obtenidas de la plataforma Kaggle. El modelo alcanzó una precisión del 98.93% en el conjunto de prueba, destacando su eficacia en la clasificación de estas imágenes.

## Contenidos del Repositorio

- **analyze.py**: Script para evaluar el modelo entrenado y generar reportes de métricas de clasificación.
- **preprocess.py**: Script para preprocesar las imágenes histopatológicas, redimensionándolas y organizándolas en directorios.
- **train_cnn.py**: Script para entrenar la CNN utilizando TensorFlow, con optimización y guardado del mejor modelo.

## Requisitos

- Python 3.6+
- TensorFlow 2.0+
- NumPy
- Matplotlib
- scikit-learn
- Seaborn
- Pillow
- tqdm

Puedes instalar los requisitos utilizando pip:

```sh
pip install tensorflow numpy matplotlib scikit-learn seaborn pillow tqdm
```

## Uso

### Preprocesamiento de Imágenes

El primer paso es preprocesar las imágenes:

```sh
python preprocess.py
```

Este script redimensionará las imágenes a 256x256 píxeles y las organizará en directorios para su posterior uso en el entrenamiento.

### Entrenamiento del Modelo

Para entrenar el modelo, utiliza el siguiente comando:

```sh
python train_cnn.py
```

El modelo se entrenará y se guardará el mejor modelo en `best_model.keras`.

### Evaluación del Modelo

Para evaluar el modelo y generar métricas de rendimiento, utiliza el siguiente comando:

```sh
python analyze.py
```

Este script cargará el modelo guardado, realizará predicciones en el conjunto de prueba y generará una matriz de confusión y un reporte de clasificación.

## Resultados

El modelo CNN alcanzó una precisión del 98.93% en el conjunto de prueba. A continuación se presenta la matriz de confusión y el reporte de clasificación:

| Clase      | Precisión | Recall | F1-score |
|------------|-----------|--------|----------|
| colon_aca  | 0.99      | 0.99   | 0.99     |
| colon_n    | 0.99      | 1.00   | 1.00     |
| lung_aca   | 0.97      | 0.98   | 0.98     |
| lung_n     | 1.00      | 1.00   | 1.00     |
| lung_scc   | 0.99      | 0.97   | 0.98     |
| **Total**  | **0.99**  | **0.99**| **0.99** |
## Datos

Los datos utilizados en este proyecto están disponibles en Kaggle: [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
