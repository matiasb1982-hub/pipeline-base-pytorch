# Pipeline base de entrenamiento, validación y clasificador

**Checkpoint 1 — Proyecto integrador de Deep Learning**

## 1. Objetivo

Implementar un pipeline reproducible en PyTorch que detecte automáticamente el dispositivo disponible, entrene un clasificador base con Adam, mida pérdida y Accuracy por época, evalúe datos separados y guarde métricas y pesos.

## 2. Estructura del repositorio

```text
pipeline-base-pytorch/
├── data/                         # Reservado para datasets externos
├── notebooks/                    # Reservado para exploración opcional
├── src/
│   └── train.py                  # Pipeline completo
├── results/
│   ├── metrics.json              # Historial y resultados de evaluación
│   └── model_state_dict.pt       # Pesos entrenados (se generan al ejecutar)
├── .gitignore
├── README.md
└── requirements.txt
```

El experimento usa datos sintéticos generados localmente; no necesita descargar archivos ni colocar datos en `data/`.

## 3. Requisitos y ejecución

Recomendado: Python 3.10 o superior.

```bash
python -m venv .venv
```

Activar el entorno:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Instalar dependencias y ejecutar:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train.py
```

El script imprime la versión de PyTorch, el dispositivo detectado y las métricas de cada época. Al terminar genera `results/metrics.json` y `results/model_state_dict.pt`.

## 4. Configuración del experimento

| Parámetro | Valor |
|---|---|
| Dataset | Sintético, clasificación multiclase |
| Observaciones | 2.400 |
| Variables de entrada | 10 |
| Clases | 3 |
| División | 70 % entrenamiento, 15 % validación, 15 % prueba |
| Arquitectura | `Linear(10, 32) → ReLU → Linear(32, 3)` |
| Función de pérdida | `CrossEntropyLoss` |
| Optimizador | Adam |
| `learning_rate` | `0.01` |
| `batch_size` | `64` |
| Épocas | `30` |
| Semilla | `42` |
| Métrica | Accuracy |

El dataset sintético se eligió para que el primer checkpoint sea ejecutable y reproducible sin depender de descargas. Las tres clases se generan alrededor de centros diferentes con ruido aleatorio controlado por una semilla.

## 5. Dispositivo y reproducibilidad

`get_device()` prioriza CUDA si hay una GPU NVIDIA compatible; si no, utiliza MPS cuando está disponible en Apple Silicon y finalmente CPU. Los tensores de cada lote se envían al mismo dispositivo que el modelo mediante `.to(device)`.

Se fijan las semillas de Python, NumPy y PyTorch. También se configura cuDNN para un comportamiento más determinista cuando corresponde. La reproducibilidad exacta puede variar entre hardware, versiones y dispositivos.

## 6. Ciclo de entrenamiento y validación

En cada lote de entrenamiento se ejecutan explícitamente estos pasos:

1. `optimizer.zero_grad()` limpia los gradientes acumulados.
2. `model(features)` ejecuta el *forward pass*.
3. `criterion(logits, labels)` calcula la pérdida.
4. `loss.backward()` calcula gradientes mediante `torch.autograd`.
5. `optimizer.step()` actualiza los parámetros con Adam.

Después de cada época, `evaluate()` calcula `val_loss` y `val_accuracy` con `model.eval()` y `torch.no_grad()`. Los datos de validación no participan en la actualización de pesos. Al finalizar, se evalúa una única vez el conjunto de prueba, reservado para estimar el rendimiento final.

## 7. Resultados e interpretación

El script registra por época:

- `train_loss`: pérdida promedio en entrenamiento.
- `val_loss`: pérdida promedio en validación.
- `val_accuracy`: proporción de predicciones correctas en validación.

Los valores reales quedan guardados en `results/metrics.json`. Para interpretar la curva, comparar las primeras y últimas épocas: una reducción de `train_loss` indica que el modelo está aprendiendo sobre entrenamiento. Si `val_loss` también baja o se mantiene baja y `val_accuracy` mejora, el aprendizaje generaliza razonablemente. Si `train_loss` continúa bajando mientras `val_loss` sube durante varias épocas, puede existir sobreajuste (*overfitting*).

No se deben inventar resultados antes de ejecutar el script: la interpretación final debe basarse en los valores registrados en `metrics.json`.

## 8. Versiones utilizadas

La versión exacta de PyTorch se imprime en consola y se escribe en `results/metrics.json` al ejecutar el experimento. `requirements.txt` define una versión mínima compatible (`torch>=2.2.0`); el entorno puede instalar una versión posterior compatible. Para congelar exactamente el entorno después de la ejecución, guardar la salida de:

```bash
python -m pip freeze > requirements-lock.txt
```

## 9. Buenas prácticas de Git

Ejemplo de comandos para versionar el trabajo:

```bash
git init
git add README.md requirements.txt src data .gitignore
git commit -m "feat: add reproducible PyTorch training pipeline"
git add results/metrics.json
git commit -m "docs: record baseline experiment metrics"
```

No es obligatorio versionar el archivo binario de pesos (`model_state_dict.pt`) para demostrar el pipeline; se puede regenerar ejecutando el script. Si se comparte, documentar su origen y la configuración usada.

## 10. Alcance y limitaciones

Este checkpoint prioriza la infraestructura técnica y no pretende ser un modelo de producción. El dataset es sintético y sirve para comprobar el ciclo completo. En la siguiente iteración puede sustituirse `make_dataset()` por la carga de MNIST, Iris u otro dataset del proyecto, conservando el ciclo de entrenamiento, validación y tracking.
