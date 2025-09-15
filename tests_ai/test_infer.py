import numpy as np
import tensorflow as tf
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import ai.server as srv


def test_predict_endpoint():
    # Parchar artefactos con modelo pequeño para velocidad
    srv.feature_map = {"columns": ["actor_level", "enemy_level"]}
    srv.label_encoder.classes_ = np.array(["BASICO", "ATAQUE", "SPECIAL"])
    srv.model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    client = TestClient(srv.app)
    payload = {
        "hero": {"type": "MACHETE_ROGUE", "level": 1, "hp": 30, "hp_pct": 1, "power": 8, "attack": 10, "defense": 8},
        "enemy": {"type": "FIRE_MAGE", "level": 1, "hp": 32, "hp_pct": 1, "power": 8, "attack": 10, "defense": 10}
    }
    resp = client.post('/predict', json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 'action' in data and 'confidence' in data
