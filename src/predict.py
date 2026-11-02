"""Predict the class of one or more MRI images, optionally with a Grad-CAM heat-map.

Usage:
    python -m src.predict --model xception --images path/to/img1.jpg path/to/img2.jpg --gradcam
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import load_config, run_dir
from .models import PRETTY


def load_image(path, size):
    import tensorflow as tf

    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    return tf.cast(tf.image.resize(img, (size, size), antialias=True), tf.float32).numpy()


def gradcam(model, image, class_idx=None):
    """Grad-CAM on the last feature map of the backbone (works with the nested backbone)."""
    import tensorflow as tf

    import keras

    base = next(l for l in model.layers if isinstance(l, keras.Model))
    head = model.layers[model.layers.index(base) + 1:]
    x = model.get_layer("scale_to_pm1")(image[None])
    with tf.GradientTape() as tape:
        feat = base(x, training=False)
        tape.watch(feat)
        y = feat
        for layer in head:
            y = layer(y, training=False)
        idx = int(tf.argmax(y[0])) if class_idx is None else class_idx
        score = y[:, idx]
    grads = tape.gradient(score, feat)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feat[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., None], image.shape[:2]).numpy()[..., 0]
    return cam, idx, y[0].numpy()


def main():
    import keras

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--gradcam", action="store_true")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    classes = cfg["data"]["classes"]
    model = keras.models.load_model(run_dir(cfg, args.model) / "best_model.keras")
    size = model.input_shape[1]
    out = run_dir(cfg, args.model) / "predictions"
    out.mkdir(exist_ok=True)

    for p in args.images:
        img = load_image(p, size)
        probs = model.predict(img[None], verbose=0)[0]
        k = int(probs.argmax())
        print(f"{p}: {classes[k]} ({probs[k]:.1%})  |  " +
              ", ".join(f"{c}={v:.3f}" for c, v in zip(classes, probs)))
        if args.gradcam:
            cam, k, _ = gradcam(model, img)
            fig, ax = plt.subplots(1, 2, figsize=(7, 3.6), dpi=130)
            ax[0].imshow(img.astype("uint8"))
            ax[0].set_title("Input")
            ax[1].imshow(img.astype("uint8"))
            ax[1].imshow(cam, cmap="jet", alpha=0.4)
            ax[1].set_title(f"Grad-CAM: {classes[k]} ({probs[k]:.1%})")
            for a in ax:
                a.axis("off")
            fig.suptitle(PRETTY.get(args.model, args.model))
            fig.tight_layout()
            fig.savefig(out / f"{Path(p).stem}_gradcam.png", bbox_inches="tight")
            plt.close(fig)
    if args.gradcam:
        print(f"Grad-CAM images saved in {out}")


if __name__ == "__main__":
    main()
