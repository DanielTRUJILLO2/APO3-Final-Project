# Clasificador de calidad de frutas

Este proyecto consiste en un sistema de visión por computador para clasificar la calidad de frutas frescas a partir de imágenes. La idea principal es que el usuario pueda cargar una foto de una fruta y el sistema estime si se encuentra en estado **bueno**, **regular** o **malo**. Además, se calcula una estimación aproximada del tamaño relativo de la fruta dentro de la imagen.

El proyecto fue desarrollado para el curso de Algoritmos y Programación III, siguiendo una metodología cercana a CRISP-DM: comprensión del problema, análisis de datos, preparación del dataset, entrenamiento de modelos, evaluación y despliegue mediante una aplicación web simple.

## Problema abordado

En mercados, supermercados y procesos agrícolas, la clasificación de frutas suele hacerse de forma manual. Esto puede ser subjetivo, lento y poco consistente, especialmente cuando se necesita separar productos por calidad o estado de madurez.

Este proyecto busca proponer una solución inicial usando modelos de machine learning y deep learning para apoyar esa clasificación de forma automática.

## Frutas utilizadas

El dataset trabajado incluye seis frutas:

- Apple
- Banana
- Guava
- Lime
- Orange
- Pomegranate

Cada fruta fue organizada en tres categorías de calidad:

- Good
- Regular
- Bad

Para el entrenamiento, estas categorías se mapearon a:

- `bueno`
- `regular`
- `malo`

## Estructura del proyecto

```text
fruit-quality-classifier/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── balanced_raw/
│   ├── splits/
│   └── metadata.csv
├── models/
│   ├── random_forest.pkl
│   ├── svm.pkl
│   └── cnn_model.keras
├── reports/
│   ├── cnn_results.csv
│   ├── ml_results.csv
│   ├── cnn_history.csv
│   └── figures/
├── src/
│   ├── prepare_dataset.py
│   ├── features.py
│   ├── train_ml.py
│   ├── train_cnn.py
│   ├── predict.py
│   └── audit_predictions.py
├── requirements.txt
└── README.md
```

## Preparación de los datos

Las imágenes originales se organizaron en carpetas según fruta y calidad. Luego se creó una versión balanceada del dataset con la misma cantidad de imágenes por combinación de fruta y calidad.

El dataset final usado para entrenamiento quedó con:

```text
1080 imágenes en total
756 imágenes para entrenamiento
162 imágenes para validación
162 imágenes para prueba
```

Las clases quedaron balanceadas en los tres conjuntos:

```text
train: 252 bueno, 252 regular, 252 malo
val:   54 bueno, 54 regular, 54 malo
test:  54 bueno, 54 regular, 54 malo
```

Para preparar el dataset se usa:

```bash
python src/prepare_dataset.py
```

Este script genera los splits y un archivo `metadata.csv` con información de cada imagen procesada.

## Modelos entrenados

Se entrenaron dos modelos tradicionales de machine learning y una red neuronal convolucional:

- Random Forest
- SVM
- CNN simple

Los modelos tradicionales usan características extraídas manualmente de las imágenes, mientras que la CNN trabaja directamente con las imágenes redimensionadas.

## Resultados obtenidos

Los resultados principales en el conjunto de prueba fueron:

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| SVM | 0.914 | 0.913 |
| Random Forest | 0.907 | 0.908 |
| CNN | 0.901 | 0.901 |

El modelo con mejor desempeño general fue **SVM**, aunque Random Forest y CNN obtuvieron resultados muy cercanos.

La categoría más difícil de clasificar fue `regular`, ya que visualmente se encuentra en un punto intermedio entre frutas buenas y frutas malas. Esto hace que algunas imágenes puedan confundirse con las otras dos clases.

## Entrenamiento

Para entrenar los modelos tradicionales:

```bash
python src/train_ml.py
```

Para entrenar la CNN:

```bash
python src/train_cnn.py
```

Los modelos entrenados se guardan en la carpeta `models/` y los resultados en `reports/`.

## Predicción individual

Se puede probar una imagen individual con:

```bash
python src/predict.py ruta_de_la_imagen svm
```

También se pueden usar:

```bash
python src/predict.py ruta_de_la_imagen random_forest
python src/predict.py ruta_de_la_imagen cnn
```

El resultado muestra:

- modelo usado
- calidad estimada
- tamaño relativo
- área relativa aproximada
- confianza de la predicción

## Aplicación web

El proyecto incluye una aplicación en Streamlit que permite subir una imagen o tomar una foto con cámara, seleccionar un modelo y ver el resultado de la predicción.

Para ejecutarla:

```bash
streamlit run app/streamlit_app.py
```

La app muestra:

- calidad estimada de la fruta
- tamaño relativo aproximado
- confianza del modelo
- explicación básica del resultado

Por defecto se usa **SVM**, ya que fue el modelo con mejor rendimiento en las pruebas.

## Instalación

Primero se recomienda crear y activar un entorno virtual:

```bash
python -m venv venv
```

En Windows:

```bash
venv\Scripts\activate
```

Luego instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Observaciones

El sistema funciona mejor cuando la imagen contiene una sola fruta, con fondo simple y buena iluminación. Si la fruta está en un estado ambiguo, especialmente entre `bueno` y `regular`, la confianza del modelo puede bajar o la predicción puede confundirse.

Este proyecto no busca reemplazar una evaluación humana experta, sino mostrar una aproximación inicial al uso de modelos de clasificación para apoyar procesos de revisión de calidad en frutas.

## Trabajo futuro

Algunas mejoras posibles serían:

- Agregar más imágenes tomadas en condiciones reales.
- Probar más frutas y más variaciones de fondo.
- Mejorar la estimación de tamaño usando segmentación más precisa.
- Implementar modelos con transferencia de aprendizaje.
- Evaluar el sistema con imágenes externas no vistas durante el entrenamiento.
- Agregar una predicción adicional para identificar el tipo de fruta.

## Referencias

- Dataset Fruit Quality Classification, Kaggle.
- Documentación oficial de Scikit-learn.
- Documentación oficial de TensorFlow/Keras.
- Documentación oficial de Streamlit.
