# Pipeline de entrenamiento, validación y clasificador base

## 1. Descripción del proyecto

Este proyecto implementa un pipeline básico de aprendizaje automático utilizando PyTorch. Su objetivo es definir, entrenar, validar y evaluar un clasificador basado en una red neuronal, registrando las métricas obtenidas durante el proceso.

## 2. Objetivos

- Configurar un entorno reproducible para el entrenamiento.
- Detectar automáticamente el dispositivo disponible (CPU o GPU compatible).
- Definir un modelo de clasificación mediante `nn.Module`.
- Entrenar el modelo utilizando el optimizador Adam y retropropagación.
- Calcular la pérdida de entrenamiento y validación.
- Registrar la precisión (*Accuracy*) en cada época.
- Evaluar el modelo con un conjunto de prueba independiente.
- Generar gráficos y guardar las métricas y los parámetros entrenados.

## 3. Tecnologías utilizadas

- Python 3.13.16 (entorno de ejecución utilizado).
- PyTorch 2.11.0+cpu (entorno de ejecución utilizado).
- NumPy.
- Matplotlib.
- Google Colab para ejecutar el experimento.
- GitHub para el control de versiones y la entrega.

## 4. Dataset

Para esta implementación se utiliza un conjunto de datos sintético generado mediante PyTorch. Contiene 2.400 ejemplos, 10 características de entrada y 3 clases.

La distribución de los datos es:

| Conjunto | Cantidad | Proporción |
|---|---:|---:|
| Entrenamiento | 1.680 | 70 % |
| Validación | 360 | 15 % |
| Prueba | 360 | 15 % |
| **Total** | **2.400** | **100 %** |

Los datos sintéticos permiten comprobar el funcionamiento del pipeline sin depender de un dataset externo. Los resultados corresponden exclusivamente a este conjunto artificial y no deben interpretarse como resultados de una aplicación con datos reales.

## 5. Arquitectura del modelo

El clasificador base está implementado como una clase que hereda de `torch.nn.Module`.

La arquitectura contiene:

1. Una capa lineal de entrada de 10 características a 32 neuronas.
2. Una función de activación ReLU.
3. Una capa lineal de salida con 3 valores, uno por clase.

La función de pérdida utilizada es `CrossEntropyLoss` y el optimizador es Adam.

## 6. Configuración del entrenamiento

| Parámetro | Valor |
|---|---|
| Semilla aleatoria | 42 |
| Épocas | 30 |
| Tamaño del batch | 64 |
| Learning rate | 0.01 |
| Optimizador | Adam |
| Función de pérdida | CrossEntropyLoss |
| Dispositivo observado | CPU |

El código establece semillas aleatorias para facilitar la reproducibilidad del experimento.

## 7. Proceso de entrenamiento y validación

En cada época se realizan los siguientes pasos:

1. Se activa el modo entrenamiento mediante `model.train()`.
2. Se reinician los gradientes con `optimizer.zero_grad()`.
3. Se calculan las predicciones del modelo.
4. Se calcula la función de pérdida.
5. Se realiza la retropropagación con `loss.backward()`.
6. Se actualizan los parámetros mediante `optimizer.step()`.
7. Se evalúa el modelo sobre el conjunto de validación sin actualizar sus parámetros.
8. Se registran la pérdida de entrenamiento, la pérdida de validación y la Accuracy de validación.

Al finalizar las 30 épocas, el modelo se evalúa sobre el conjunto de prueba, que se mantiene separado del entrenamiento y la validación.

## 8. Resultados generados

La ejecución produce los siguientes archivos:

- `results/loss_curve.png`: evolución de la pérdida de entrenamiento y validación.
- `results/accuracy_curve.png`: evolución de la Accuracy de validación.
- `results/metrics.json`: configuración, historial de métricas y resultados finales.
- `results/model_weights.pth`: parámetros aprendidos por el modelo.

Los valores finales de pérdida y precisión deben consultarse en `results/metrics.json`, que contiene los resultados de la ejecución.

## 9. Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── pipeline_entrenamiento.ipynb
├── src/
│   └── train.py
└── results/
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── metrics.json
    └── model_weights.pth
```

La estructura representa los archivos principales esperados. Si algún archivo todavía no está presente en el repositorio, deberá incorporarse antes de la entrega.

## 10. Instalación y ejecución

Se recomienda utilizar un entorno virtual con una versión de Python compatible con las dependencias instaladas.

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Para ejecutar el experimento en Google Colab, abrir `pipeline_entrenamiento.ipynb` y ejecutar las celdas en orden.

Si se utiliza el script `src/train.py`, verificar que contenga la implementación correspondiente al mismo experimento.

## 11. Conclusión

El proyecto desarrolla un pipeline básico de clasificación con PyTorch, incluyendo configuración reproducible, separación de datos, entrenamiento, validación, evaluación y almacenamiento de resultados.

La implementación constituye una base para incorporar posteriormente datasets reales, ampliar la arquitectura y comparar distintas configuraciones de entrenamiento.
