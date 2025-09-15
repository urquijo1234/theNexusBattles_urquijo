"""Servicio de inferencia FastAPI para seleccionar acciones."""
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Carga de artefactos
def _load_artifacts():
    """Carga artefactos si existen, de lo contrario usa placeholders.

    Permite importar el módulo sin archivos pesados; los tests pueden parchear
    `model`, `label_encoder` y `feature_map` posteriormente.
    """
    models_dir = Path(__file__).parent / "models"
    try:
        model = tf.keras.models.load_model(models_dir / "hero_action_selector.keras")
        with open(models_dir / "label_encoder.pkl", "rb") as fh:
            le = pickle.load(fh)
        with open(models_dir / "feature_map.json", "r", encoding="utf-8") as fh:
            feature_map = json.load(fh)
    except Exception:
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(0,)),
                                     tf.keras.layers.Dense(3, activation="softmax")])
        le = LabelEncoder()
        le.fit(["BASICO", "ATAQUE", "SPECIAL"])
        feature_map = {"columns": []}
    return model, le, feature_map


model, label_encoder, feature_map = _load_artifacts()
app = FastAPI(title="AI Action Selector")


class Stats(BaseModel):
    type: str
    level: float
    hp: float
    hp_pct: float
    power: float
    attack: float
    defense: float


class TurnState(BaseModel):
    hero: Stats
    enemy: Stats
    context: dict | None = None


def _build_features(hero: Stats, enemy: Stats) -> np.ndarray:
    values = {}
    for col in feature_map["columns"]:
        if col.startswith("actor_type_"):
            t = col.replace("actor_type_", "")
            values[col] = 1.0 if hero.type == t else 0.0
        elif col.startswith("enemy_type_"):
            t = col.replace("enemy_type_", "")
            values[col] = 1.0 if enemy.type == t else 0.0
        elif col.startswith("actor_"):
            key = col.replace("actor_", "")
            values[col] = getattr(hero, key, 0)
        elif col.startswith("enemy_"):
            key = col.replace("enemy_", "")
            values[col] = getattr(enemy, key, 0)
        else:
            values[col] = 0
    ordered = [values[c] for c in feature_map["columns"]]
    return np.array([ordered], dtype=np.float32)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(state: TurnState):
    feats = _build_features(state.hero, state.enemy)
    preds = model(feats).numpy()[0]
    idx = int(np.argmax(preds))
    action = label_encoder.inverse_transform([idx])[0]
    return {"action": action, "confidence": float(preds[idx])}
