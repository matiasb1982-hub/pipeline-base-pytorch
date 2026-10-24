"""
Pipeline base de entrenamiento y validación en PyTorch.
Dataset sintético de clasificación multiclase: reproducible, no requiere descargas.
"""
import json
import random
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

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
RESULTS_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Fija semillas para mejorar la reproducibilidad del experimento."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Prioridad: CUDA, Apple Silicon MPS y, por último, CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset():
    """Genera datos sintéticos con tres clases separables y algo de ruido."""
    generator = torch.Generator().manual_seed(SEED)
    centers = torch.tensor([
        [-2.0] * N_FEATURES,
        [0.0] * N_FEATURES,
        [2.0] * N_FEATURES,
    ])
    labels = torch.randint(0, N_CLASSES, (N_SAMPLES,), generator=generator)
    features = centers[labels] + 0.9 * torch.randn(
        N_SAMPLES, N_FEATURES, generator=generator
    )
    # Mezcla el orden de las observaciones de forma reproducible.
    permutation = torch.randperm(N_SAMPLES, generator=generator)
    features, labels = features[permutation], labels[permutation]

    train_end = int(N_SAMPLES * TRAIN_RATIO)
    val_end = train_end + int(N_SAMPLES * VAL_RATIO)
    x_train, y_train = features[:train_end], labels[:train_end]
    x_val, y_val = features[train_end:val_end], labels[train_end:val_end]
    x_test, y_test = features[val_end:], labels[val_end:]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class BaseClassifier(nn.Module):
    """MLP pequeño para clasificación multiclase."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def evaluate(model, loader, criterion, device):
    """Calcula loss promedio y accuracy sobre datos separados."""
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
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def main():
    set_seed()
    device = get_device()
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Dispositivo seleccionado: {device}")

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_dataset()
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)

    model = BaseClassifier(N_FEATURES, N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()          # Evita acumular gradientes.
            logits = model(features)        # Forward pass.
            loss = criterion(logits, labels)  # Cálculo de pérdida.
            loss.backward()                 # Backpropagation / autograd.
            optimizer.step()                # Actualización con Adam.

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

        train_loss = running_loss / seen
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 6),
        }
        history.append(row)
        print(
            f"Época {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_accuracy={val_accuracy:.4f}"
        )

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    results = {
        "seed": SEED,
        "device": str(device),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
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
        "history": history,
        "test": {
            "loss": round(test_loss, 6),
            "accuracy": round(test_accuracy, 6),
        },
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    torch.save(model.state_dict(), RESULTS_DIR / "model_state_dict.pt")
    print("\\nResultado final en conjunto de prueba:")
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2%}")
    print(f"Métricas guardadas en: {RESULTS_DIR / 'metrics.json'}")
    print(f"Pesos guardados en: {RESULTS_DIR / 'model_state_dict.pt'}")


if __name__ == "__main__":
    main()
