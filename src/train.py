"""
train.py
────────
End-to-end training and evaluation script for the Radar Signal CNN.

Usage (from project root)
────────────────────────
    python src/train.py                        # default settings
    python src/train.py --epochs 50 --lr 5e-4
    python src/train.py --model lite --batch 64
    python src/train.py --no-denoise --seed 0

CLI arguments
─────────────
  --epochs          int   (default 30)
  --lr              float (default 1e-3)
  --batch           int   (default 32)
  --model           str   standard|lite|deep (default standard)
  --seed            int   (default 42)
  --no-denoise            disable Wiener denoising
  --samples         int   samples per class (default 200)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── local imports (works when run as `python src/train.py` from project root)
sys.path.insert(0, str(Path(__file__).parent))

from preprocess   import (
    generate_synthetic_dataset, preprocess_signals,
    split_dataset, RadarSignalDataset, SIGNAL_CLASSES, NUM_CLASSES,
)
from spectrogram  import (
    batch_to_spectrograms, plot_class_grid,
    plot_single_spectrogram, plot_stft_components,
)
from model        import get_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR     = Path("results")
SPEC_SAMPLE_DIR = RESULTS_DIR / "spectrogram_samples"
METRICS_DIR     = RESULTS_DIR / "model_metrics"


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _dark_style() -> None:
    plt.rcParams.update({
        "axes.facecolor":    "#12122a",
        "figure.facecolor":  "#0d0d1a",
        "axes.edgecolor":    "#2a2a4a",
        "axes.labelcolor":   "#c0c0d0",
        "xtick.color":       "#888899",
        "ytick.color":       "#888899",
        "text.color":        "#e0e0ff",
        "grid.color":        "#1e1e38",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
    })


def plot_training_history(
    train_losses: list,
    val_losses:   list,
    train_accs:   list,
    val_accs:     list,
    save_path:    Path,
) -> None:
    _dark_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", color="#e0e0ff", fontsize=14, fontweight="bold")

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, color="#4a9eff", linewidth=1.8, label="Train Loss")
    ax1.plot(epochs, val_losses,   color="#ff6b6b", linewidth=1.8,
             linestyle="--", label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Cross-Entropy Loss", color="#e0e0ff")
    ax1.legend(facecolor="#12122a", edgecolor="#2a2a4a")
    ax1.grid(True)

    ax2.plot(epochs, [a * 100 for a in train_accs],
             color="#51cf66", linewidth=1.8, label="Train Acc")
    ax2.plot(epochs, [a * 100 for a in val_accs],
             color="#ffd43b", linewidth=1.8, linestyle="--", label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Classification Accuracy", color="#e0e0ff")
    ax2.legend(facecolor="#12122a", edgecolor="#2a2a4a")
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    logger.info("Saved training history → %s", save_path)
    plt.close(fig)


def plot_confusion_matrix_custom(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list,
    save_path:   Path,
) -> None:
    _dark_style()
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrices", color="#e0e0ff", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Counts", "Normalised (row %)"],
    ):
        sns.heatmap(
            data, ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues",
            annot=True,
            fmt=fmt,
            linewidths=0.5,
            linecolor="#1a1a2e",
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, color="#e0e0ff", fontsize=11)
        ax.set_xlabel("Predicted", color="#c0c0d0")
        ax.set_ylabel("True",      color="#c0c0d0")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0d0d1a")
    logger.info("Saved confusion matrices → %s", save_path)
    plt.close(fig)


def plot_per_class_accuracy(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list,
    save_path:   Path,
) -> None:
    _dark_style()
    per_class = [
        accuracy_score(y_true[y_true == i], y_pred[y_true == i])
        for i in range(len(class_names))
    ]
    colors = ["#4a9eff", "#ff6b6b", "#51cf66",
              "#ffd43b", "#cc5de8", "#ff922b"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(class_names, [a * 100 for a in per_class],
                   color=colors[:len(class_names)], edgecolor="#0d0d1a",
                   linewidth=0.8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Classification Accuracy", color="#e0e0ff",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 110)
    for bar, acc in zip(bars, per_class):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{acc * 100:.1f}%", va="center", ha="left",
                color="#e0e0ff", fontsize=9)
    ax.axvline(80, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    logger.info("Saved per-class accuracy → %s", save_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimiser:  torch.optim.Optimizer,
    device:     torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimiser.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        logits = model(specs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def predict(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for specs, labels in loader:
        specs = specs.to(device)
        logits = model(specs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Radar Signal CNN – Training Script")
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--model",     type=str,   default="standard",
                   choices=["standard", "lite", "deep"])
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--no-denoise", action="store_true")
    p.add_argument("--samples",   type=int,   default=200,
                   help="Samples per class")
    return p.parse_args()


def main():
    args = parse_args()

    # ── reproducibility ──────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    SPEC_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate dataset ──────────────────────────────────────────────
    logger.info("=== Step 1: Generating Synthetic Radar Dataset ===")
    signals, labels = generate_synthetic_dataset(
        n_samples_per_class=args.samples,
        random_seed=args.seed,
    )
    logger.info("Dataset: %d signals, %d classes", len(signals), NUM_CLASSES)

    # ── 2. Preprocess ────────────────────────────────────────────────────
    logger.info("=== Step 2: Preprocessing Signals ===")
    signals = preprocess_signals(signals, apply_denoising=not args.no_denoise)

    # ── 3. Visualise spectrograms ────────────────────────────────────────
    logger.info("=== Step 3: Generating Spectrogram Visualisations ===")
    # Per-class sample grid
    plot_class_grid(
        signals, labels, SIGNAL_CLASSES,
        n_per_class=3,
        save_path=SPEC_SAMPLE_DIR / "class_overview_grid.png",
    )
    # One detailed panel per class
    for cls_idx, cls_name in SIGNAL_CLASSES.items():
        cls_signals = signals[labels == cls_idx]
        if len(cls_signals) == 0:
            continue
        plot_single_spectrogram(
            cls_signals[0], cls_name, cls_idx,
            save_path=SPEC_SAMPLE_DIR / f"class_{cls_idx}_{cls_name.replace(' ', '_').replace('–','')}.png",
        )
        # STFT component plot for first two classes only (avoid long CI runs)
        if cls_idx < 2:
            plot_stft_components(
                cls_signals[0], cls_name,
                save_path=SPEC_SAMPLE_DIR / f"stft_components_class_{cls_idx}.png",
            )
    logger.info("Spectrogram visualisations saved to %s", SPEC_SAMPLE_DIR)

    # ── 4. Compute spectrograms ──────────────────────────────────────────
    logger.info("=== Step 4: Computing Spectrograms for All Signals ===")
    spectrograms = batch_to_spectrograms(signals)   # (N, 1, F, T)
    logger.info("Spectrogram batch shape: %s", spectrograms.shape)

    # ── 5. Split ─────────────────────────────────────────────────────────
    logger.info("=== Step 5: Splitting Dataset ===")
    (X_train, X_val, X_test,
     y_train, y_val,  y_test) = split_dataset(spectrograms, labels)

    train_ds = RadarSignalDataset(X_train, y_train, augment=True)
    val_ds   = RadarSignalDataset(X_val,   y_val)
    test_ds  = RadarSignalDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0)

    # ── 6. Build model ───────────────────────────────────────────────────
    logger.info("=== Step 6: Building CNN Model ===")
    model = get_model(variant=args.model, num_classes=NUM_CLASSES).to(device)
    logger.info("Parameters: %s", f"{model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-5)

    # ── 7. Training loop ─────────────────────────────────────────────────
    logger.info("=== Step 7: Training (%d epochs) ===", args.epochs)
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0
    best_path    = METRICS_DIR / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimiser, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);    val_accs.append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | TrainLoss=%.4f TrainAcc=%.2f%% "
            "| ValLoss=%.4f ValAcc=%.2f%% | %.1fs",
            epoch, args.epochs,
            tr_loss, tr_acc * 100,
            va_loss, va_acc * 100,
            elapsed,
        )

    # ── 8. Evaluate on test set ──────────────────────────────────────────
    logger.info("=== Step 8: Evaluating on Test Set ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_preds, test_labels = predict(model, test_loader, device)

    test_acc = accuracy_score(test_labels, test_preds)
    logger.info("Test Accuracy: %.2f%%", test_acc * 100)

    class_names = [SIGNAL_CLASSES[i] for i in range(NUM_CLASSES)]
    report = classification_report(test_labels, test_preds, target_names=class_names)
    logger.info("\nClassification Report:\n%s", report)

    # Save text report
    report_path = METRICS_DIR / "classification_report.txt"
    report_path.write_text(
        f"Test Accuracy: {test_acc * 100:.2f}%\n\n{report}"
    )

    # ── 9. Save visualisations ───────────────────────────────────────────
    logger.info("=== Step 9: Saving Result Visualisations ===")
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=METRICS_DIR / "training_history.png",
    )
    plot_confusion_matrix_custom(
        test_labels, test_preds, class_names,
        save_path=METRICS_DIR / "confusion_matrix.png",
    )
    plot_per_class_accuracy(
        test_labels, test_preds, class_names,
        save_path=METRICS_DIR / "per_class_accuracy.png",
    )

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Best Val Accuracy : %.2f%%", best_val_acc * 100)
    logger.info("  Test Accuracy     : %.2f%%", test_acc * 100)
    logger.info("  Model saved       : %s",    best_path)
    logger.info("  Results dir       : %s",    RESULTS_DIR)
    logger.info("=" * 60)

    return {
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "train_accs":   train_accs,
        "val_accs":     val_accs,
    }


if __name__ == "__main__":
    main()