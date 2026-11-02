"""Evaluation metrics and figures (confusion matrix, training curves, ROC)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, probs, classes: list[str]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    report = classification_report(y_true, y_pred, labels=range(len(classes)),
                                   target_names=classes, digits=4, output_dict=True, zero_division=0)
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per_class[c] = {
            "precision": report[c]["precision"],
            "recall_sensitivity": report[c]["recall"],
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "f1": report[c]["f1-score"],
            "support": int(cm[i].sum()),
        }
    try:
        auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_auc_ovr": auc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=range(len(classes)), target_names=classes, digits=4, zero_division=0),
    }


def plot_confusion(cm, classes, title, path, cmap="Blues"):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=150)
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), [c.capitalize() for c in classes], rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), [c.capitalize() for c in classes])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    ep = np.arange(1, len(history["loss"]) + 1)
    axes[0].plot(ep, history["accuracy"], label="train")
    axes[0].plot(ep, history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[1].plot(ep, history["loss"], label="train")
    axes[1].plot(ep, history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    if "stage_boundary" in history:
        for a in axes:
            a.axvline(history["stage_boundary"] + 0.5, ls="--", c="gray", lw=1)
            a.text(history["stage_boundary"] + 0.7, a.get_ylim()[1], "fine-tuning →",
                   va="top", fontsize=8, color="gray")
    for a in axes:
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, probs, classes, path, title):
    fig, ax = plt.subplots(figsize=(5, 4.6), dpi=150)
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(np.asarray(y_true) == i, probs[:, i])
        try:
            auc = roc_auc_score(np.asarray(y_true) == i, probs[:, i])
        except ValueError:
            auc = float("nan")
        ax.plot(fpr, tpr, label=f"{c.capitalize()} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_all(out_dir: Path, name: str, y_true, probs, classes) -> dict:
    m = compute_metrics(y_true, probs, classes)
    (out_dir / "metrics.json").write_text(json.dumps({k: v for k, v in m.items()}, indent=2))
    (out_dir / "classification_report.txt").write_text(m["classification_report"])
    plot_confusion(m["confusion_matrix"], classes, f"{name} — confusion matrix",
                   out_dir / "confusion_matrix.png")
    plot_roc(y_true, probs, classes, out_dir / "roc_curves.png", f"{name} — ROC (one-vs-rest)")
    print(f"\n===== {name} (test set) =====")
    print(m["classification_report"])
    print(f"Accuracy {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"macro-AUC {m['macro_auc_ovr']:.4f} | kappa {m['cohen_kappa']:.4f}")
    return m
