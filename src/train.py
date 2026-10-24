
"""
Pipeline base de entrenamiento, validación y clasificación con PyTorch.
Dataset sintético reproducible de clasificación multiclase.
"""

import json
import random
import platform
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


# --------------------------------------------------
# 1. Configuración
# --------------------------------------------------

SEED = 42
LEARNING_RATE = 0.01
BATCH_SIZE = 64
EPOCHS = 30
N_SAMPLES = 2400
N_FEATURES = 10
N_CLASSES = 3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    """Fija semillas para favorecer la reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Detecta CUDA, MPS o CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# --------------------------------------------------
# 2. Dataset sintético
# --------------------------------------------------

def make_dataset():
    """Genera 2400 ejemplos, 10 características y 3 clases."""
    generator = torch.Generator().manual_seed(SEED)

    centers = torch.tensor([
        [-2.0] * N_FEATURES,
        [0.0] * N_FEATURES,
        [2.0] * N_FEATURES,
    ])

    labels = torch.randint(
        0, N_CLASSES, (N_SAMPLES,), generator=generator
    )

    features = centers[labels] + 0.9 * torch.randn(
        N_SAMPLES, N_FEATURES, generator=generator
    )

    permutation = torch.randperm(
        N_SAMPLES, generator=generator
    )
    features = features[permutation]
    labels = labels[permutation]

    train_end = int(N_SAMPLES * TRAIN_RATIO)
    val_end = train_end + int(N_SAMPLES * VAL_RATIO)

    return (
        (features[:train_end], labels[:train_end]),
        (features[train_end:val_end], labels[train_end:val_end]),
        (features[val_end:], labels[val_end:]),
    )


# --------------------------------------------------
# 3. Modelo base
# --------------------------------------------------

class BaseClassifier(nn.Module):
    """Red neuronal pequeña para clasificación multiclase."""

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# 4. Evaluación
# --------------------------------------------------

def evaluate(model, loader, criterion, device):
    """Calcula pérdida promedio y Accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_examples += batch_size

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


# --------------------------------------------------
# 5. Entrenamiento, validación y prueba
# --------------------------------------------------

def main():
    set_seed()
    device = get_device()

    print("PIPELINE DE ENTRENAMIENTO")
    print("-------------------------")
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("Dispositivo:", device)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = (
        make_dataset()
    )

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = BaseClassifier(N_FEATURES, N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

        train_loss = running_loss / seen
        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Época {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2%}"
        )

    # Evaluación final sobre el conjunto de prueba
    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )

    print("\nRESULTADOS DEL CONJUNTO DE PRUEBA")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")

    # Guardar métricas
    results = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": str(device),
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "dataset": {
            "type": "synthetic_multiclass",
            "samples": N_SAMPLES,
            "features": N_FEATURES,
            "classes": N_CLASSES,
            "train": len(x_train),
            "validation": len(x_val),
            "test": len(x_test),
        },
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "history": history,
    }

    with open(
        RESULTS_DIR / "metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    # Guardar pesos del modelo
    torch.save(
        model.state_dict(),
        RESULTS_DIR / "model_weights.pth",
    )

    # Gráfico de pérdida
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs_range, history["train_loss"],
        label="Entrenamiento"
    )
    plt.plot(
        epochs_range, history["val_loss"],
        label="Validación"
    )
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Evolución de la pérdida")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
    plt.close()

    # Gráfico de Accuracy
    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs_range,
        [value * 100 for value in history["val_accuracy"]],
        label="Validation Accuracy",
    )
    plt.xlabel("Época")
    plt.ylabel("Accuracy (%)")
    plt.title("Precisión del modelo por época")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=150)
    plt.close()

    print("\nArchivos generados en results/:")
    for filename in [
        "metrics.json",
        "model_weights.pth",
        "loss_curve.png",
        "accuracy_curve.png",
    ]:
        print("-", filename)


if __name__ == "__main__":
    main()
