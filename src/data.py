"""Dataset discovery, stratified splitting and tf.data pipelines.

The Kaggle dataset can come with different folder layouts (Training/Testing,
class folders with slightly different names, ...). Instead of relying on one
fixed layout, every image is assigned to a class by looking at the names of the
folders above it. Images whose folders do not match any class are skipped and
reported, so nothing is silently mixed in.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# folder-name keywords -> class name (checked on lower-cased folder names)
CLASS_KEYWORDS = {
    "glioma": ["glioma"],
    "meningioma": ["meningioma"],
    "pituitary": ["pituitary"],
    "normal": ["normal", "notumor", "no_tumor", "no tumor", "no-tumor", "healthy"],
}
EXACT_NORMAL_NAMES = {"no", "none"}


def _class_from_path(path: Path, root: Path, classes: list[str]) -> str | None:
    for folder in reversed(path.relative_to(root).parts[:-1]):
        name = folder.lower().strip()
        if name in EXACT_NORMAL_NAMES and "normal" in classes:
            return "normal"
        for cls in classes:
            if any(k in name for k in CLASS_KEYWORDS.get(cls, [cls])):
                return cls
    return None


def scan_dataset(root: str | Path, classes: list[str]) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder '{root}' not found. Run: bash scripts/download_data.sh"
        )
    rows, skipped = [], {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS or not p.is_file():
            continue
        cls = _class_from_path(p, root, classes)
        if cls is None:
            key = str(p.parent.relative_to(root))
            skipped[key] = skipped.get(key, 0) + 1
            continue
        rows.append({"path": str(p), "label": cls})
    if skipped:
        print("[data] Skipped folders that do not match a class:")
        for k, v in sorted(skipped.items()):
            print(f"        {k}: {v} images")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No labelled images found under {root}")
    df["label_id"] = df["label"].map({c: i for i, c in enumerate(classes)})
    return df


def make_splits(cfg: dict, force: bool = False) -> dict[str, pd.DataFrame]:
    """Create (or reuse) one fixed stratified split shared by all models."""
    split_dir = Path(cfg["output_dir"]) / "splits"
    files = {s: split_dir / f"{s}.csv" for s in ("train", "val", "test")}
    if not force and all(f.exists() for f in files.values()):
        return {s: pd.read_csv(f) for s, f in files.items()}

    d = cfg["data"]
    df = scan_dataset(d["root"], d["classes"])
    train_val, test = train_test_split(
        df, test_size=d["test_size"], stratify=df["label"], random_state=cfg["seed"]
    )
    val_frac = d["val_size"] / (1.0 - d["test_size"])
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=train_val["label"], random_state=cfg["seed"]
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    out = {"train": train, "val": val, "test": test}
    for s, part in out.items():
        part.reset_index(drop=True).to_csv(files[s], index=False)
    summary = pd.DataFrame({s: p["label"].value_counts() for s, p in out.items()}).fillna(0)
    summary["total"] = summary.sum(axis=1)
    print("[data] Split summary:\n" + summary.astype(int).to_string())
    return {s: p.reset_index(drop=True) for s, p in out.items()}


def class_weights(train_df: pd.DataFrame, n_classes: int) -> dict[int, float]:
    counts = np.bincount(train_df["label_id"], minlength=n_classes)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


# ----------------------------------------------------------------------------- tf.data

def _augmenter(aug: dict):
    import keras

    layers = []
    if aug.get("flip"):
        layers.append(keras.layers.RandomFlip("horizontal"))
    if aug.get("rotation"):
        layers.append(keras.layers.RandomRotation(aug["rotation"], fill_mode="constant"))
    if aug.get("translation"):
        t = aug["translation"]
        layers.append(keras.layers.RandomTranslation(t, t, fill_mode="constant"))
    if aug.get("zoom"):
        layers.append(keras.layers.RandomZoom(aug["zoom"], fill_mode="constant"))
    if aug.get("shear") and hasattr(keras.layers, "RandomShear"):
        s = aug["shear"]
        layers.append(keras.layers.RandomShear(x_factor=s, y_factor=s, fill_mode="constant"))
    if aug.get("brightness"):
        layers.append(keras.layers.RandomBrightness(aug["brightness"], value_range=(0, 255)))
    if aug.get("contrast"):
        layers.append(keras.layers.RandomContrast(aug["contrast"]))
    return keras.Sequential(layers, name="augmentation")


def make_dataset(df: pd.DataFrame, cfg: dict, batch_size: int, training: bool):
    """Images are returned as float32 in [0, 255]; scaling happens inside the model."""
    import tensorflow as tf

    size = cfg["data"]["img_size"]

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size), antialias=True)
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label_id"].values))
    if training:
        ds = ds.shuffle(len(df), seed=cfg["seed"], reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if training:
        aug = _augmenter(cfg["augment"])
        noise = float(cfg["augment"].get("noise_std") or 0.0) * 127.5

        def augment(x, y):
            x = aug(x, training=True)
            if noise > 0:
                x = x + tf.random.normal(tf.shape(x), stddev=noise)
            return tf.clip_by_value(x, 0.0, 255.0), y

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    from .config import load_config

    ap = argparse.ArgumentParser(description="Scan the dataset and create the fixed split.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--force", action="store_true", help="re-create the split")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()
    make_splits(load_config(args.config, args.set), force=args.force)
