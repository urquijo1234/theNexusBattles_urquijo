import json
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Campos numéricos a extraer del actor y enemigo
NUMERIC_FIELDS = [
    "level", "hp", "hp_pct", "power", "attack", "defense"
]


def _flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    actor = rec.get("actor", {})
    enemy = rec.get("enemy", {})
    out = {
        "actor_type": actor.get("type", "UNKNOWN"),
        "enemy_type": enemy.get("type", "UNKNOWN"),
    }
    for f in NUMERIC_FIELDS:
        out[f"actor_{f}"] = actor.get(f, 0)
        out[f"enemy_{f}"] = enemy.get(f, 0)
    # Etiqueta
    label = rec.get("chosen_action_kind", "BASIC_ATTACK")
    if label == "BASIC_ATTACK":
        out["label"] = "BASICO"
    elif label == "SPECIAL_SKILL":
        out["label"] = "SPECIAL"
    else:
        # MASTER_SKILL y otros se agrupan como SPECIAL
        out["label"] = "SPECIAL"
    return out


def load_dataset(path: str, test_size: float = 0.2, val_size: float = 0.1,
                  artifact_path: str = "ai/artifacts/feature_map.json") -> Tuple:
    """Carga NDJSON y retorna splits + metadata.

    Guarda un JSON con el orden de columnas para la inferencia.
    """
    p = Path(path)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(_flatten_record(json.loads(line)))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)

    labels = df.pop("label")
    # One-hot para tipos
    df = pd.get_dummies(df, columns=["actor_type", "enemy_type"], prefix=["actor_type", "enemy_type"])
    # Asegurar que todas las columnas sean numéricas (float32) para evitar dtype=object
    df = df.apply(pd.to_numeric).astype(float)

    feature_map = {"columns": df.columns.tolist()}
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(feature_map, fh, indent=2)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_tmp, y_train, y_tmp = train_test_split(df.values, y, test_size=test_size, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_map, le)


__all__ = ["load_dataset"]
