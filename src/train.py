"""Two-stage transfer-learning training for one backbone.

Stage 1  backbone frozen, only the new head is trained (higher learning rate)
Stage 2  top part of the backbone is unfrozen and fine-tuned (low learning rate,
         ReduceLROnPlateau + EarlyStopping that restores the best weights)

Usage:
    python -m src.train --model xception
    python -m src.train --model inception_v3 --set train.batch_size=16
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .config import backbone_config, load_config, run_dir, set_seed
from .data import class_weights, make_dataset, make_splits
from .metrics import plot_history, save_all
from .models import BACKBONES, PRETTY, build_model, unfreeze_top


def _compile(model, lr, label_smoothing):
    import keras

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )


def _onehot(ds, n):
    import tensorflow as tf

    return ds.map(lambda x, y: (x, tf.one_hot(y, n)))


def train_backbone(cfg: dict, name: str) -> dict:
    import keras

    set_seed(cfg["seed"])
    tcfg = backbone_config(cfg, name)
    if tcfg.get("mixed_precision"):
        keras.mixed_precision.set_global_policy("mixed_float16")

    classes = cfg["data"]["classes"]
    n = len(classes)
    out = run_dir(cfg, name)
    splits = make_splits(cfg)

    bs = tcfg["batch_size"]
    train_ds = _onehot(make_dataset(splits["train"], cfg, bs, training=True), n)
    val_ds = _onehot(make_dataset(splits["val"], cfg, bs, training=False), n)
    cw = class_weights(splits["train"], n) if tcfg.get("class_weights") else None
    if cw:
        print("[train] class weights:", {classes[k]: round(v, 3) for k, v in cw.items()})

    model, base = build_model(name, cfg["data"]["img_size"], n, cfg["model"])
    ckpt = out / "best_model.keras"
    t0 = time.time()

    # ---------------- Stage 1: train the head
    _compile(model, tcfg["head_lr"], tcfg["label_smoothing"])
    print(f"\n[train] {PRETTY[name]} — stage 1: head only ({tcfg['head_epochs']} epochs)")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=tcfg["head_epochs"],
                   class_weight=cw, verbose=2)

    # ---------------- Stage 2: fine-tune the top of the backbone
    n_train = unfreeze_top(base, tcfg["unfreeze_fraction"])
    print(f"\n[train] stage 2: fine-tuning {n_train}/{len(base.layers)} backbone layers "
          f"(lr={tcfg['finetune_lr']})")
    _compile(model, tcfg["finetune_lr"], tcfg["label_smoothing"])
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=tcfg["early_stopping_patience"],
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                          patience=tcfg["reduce_lr_patience"], min_lr=tcfg["min_lr"]),
        keras.callbacks.CSVLogger(out / "finetune_log.csv"),
    ]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=tcfg["finetune_epochs"],
                   class_weight=cw, callbacks=callbacks, verbose=2)

    history = {k: list(map(float, h1.history[k] + h2.history[k]))
               for k in ("accuracy", "val_accuracy", "loss", "val_loss")}
    history["stage_boundary"] = len(h1.history["loss"])
    (out / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, out / "training_curves.png", f"{PRETTY[name]} — training curves")

    # ---------------- Predictions (ordered, no augmentation) for evaluation + ensemble
    for split in ("val", "test"):
        ds = make_dataset(splits[split], cfg, bs, training=False)
        probs = model.predict(ds, verbose=0)
        np.save(out / f"probs_{split}.npy", probs)
        np.save(out / f"labels_{split}.npy", splits[split]["label_id"].values)

    y_test = splits["test"]["label_id"].values
    metrics = save_all(out, PRETTY[name], y_test, np.load(out / "probs_test.npy"), classes)
    metrics["train_minutes"] = round((time.time() - t0) / 60, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(BACKBONES) + ["all"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.batch_size=16")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    names = list(BACKBONES) if args.model == "all" else [args.model]
    for name in names:
        train_backbone(cfg, name)


if __name__ == "__main__":
    main()
