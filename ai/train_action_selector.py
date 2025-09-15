"""Entrenamiento de selector de acciones con Keras 3."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from ai.data.pve_loader import load_dataset


def train(data_path: str, out_dir: str):
    X_train, X_val, X_test, y_train, y_val, y_test, feature_map, le = load_dataset(data_path)

    num_classes = len(le.classes_)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Pesos de clase
    classes = np.unique(y_train)
    if len(classes) > 1:
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(c): w for c, w in zip(classes, cw)}
    else:
        class_weight = {int(classes[0]): 1.0}

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, class_weight=class_weight, verbose=0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "hero_action_selector.keras")
    with open(out / "label_encoder.pkl", "wb") as fh:
        pickle.dump(le, fh)

    try:
        import tf2onnx
        onnx_model, _ = tf2onnx.convert.from_keras(model)
        with open(out / "hero_action_selector.onnx", "wb") as fh:
            fh.write(onnx_model.SerializeToString())
    except Exception:
        pass

    with open(out / "feature_map.json", "w", encoding="utf-8") as fh:
        json.dump(feature_map, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="ai/models")
    args = parser.parse_args()
    train(args.data, args.out)
