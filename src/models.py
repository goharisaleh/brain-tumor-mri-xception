"""Transfer-learning models with a customised classification head."""
from __future__ import annotations

import keras
from keras import layers, regularizers

BACKBONES = {
    "inception_resnet_v2": keras.applications.InceptionResNetV2,
    "inception_v3": keras.applications.InceptionV3,
    "xception": keras.applications.Xception,
}

PRETTY = {
    "inception_resnet_v2": "InceptionResNetV2",
    "inception_v3": "InceptionV3",
    "xception": "Xception",
    "ensemble_soft_voting": "Ensemble (soft voting)",
    "ensemble_stacking": "Ensemble (stacking)",
}


def build_model(name: str, img_size: int, n_classes: int, mcfg: dict) -> tuple[keras.Model, keras.Model]:
    """Return (full_model, backbone).

    Input images are expected in [0, 255]. All three backbones were trained on
    ImageNet with inputs scaled to [-1, 1], so the same scaling is done here
    inside the model (no separate preprocessing needed at inference time).
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(BACKBONES)}")

    base = BACKBONES[name](include_top=False, weights=mcfg.get("weights"),
                           input_shape=(img_size, img_size, 3))
    base.trainable = False

    reg = regularizers.l2(mcfg["l2"]) if mcfg.get("l2") else None
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_to_pm1")(inputs)
    x = base(x, training=False)          # keeps BatchNorm statistics frozen during fine-tuning
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(mcfg["dropout"], name="head_dropout_1")(x)
    x = layers.Dense(mcfg["dense_units"], activation="relu", kernel_regularizer=reg, name="head_dense")(x)
    x = layers.Dropout(mcfg["dropout"] / 2, name="head_dropout_2")(x)
    outputs = layers.Dense(n_classes, activation="softmax", dtype="float32", name="probs")(x)
    return keras.Model(inputs, outputs, name=name), base


def unfreeze_top(base: keras.Model, fraction: float) -> int:
    """Unfreeze the top `fraction` of backbone layers; BatchNorm layers stay frozen."""
    base.trainable = True
    n = len(base.layers)
    cut = int(n * (1.0 - fraction))
    for i, layer in enumerate(base.layers):
        trainable = i >= cut and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = trainable
    return sum(1 for layer in base.layers if layer.trainable)
