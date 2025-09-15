import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.data.pve_loader import load_dataset


def test_load_dataset(tmp_path):
    data_file = tmp_path / 'sample.ndjson'
    sample = {"actor": {"type": "MACHETE_ROGUE", "level":1, "hp":30, "hp_pct":1, "power":8, "attack":10, "defense":8},
              "enemy": {"type": "FIRE_MAGE", "level":1, "hp":32, "hp_pct":1, "power":8, "attack":10, "defense":10},
              "chosen_action_kind": "BASIC_ATTACK"}
    with data_file.open('w', encoding='utf-8') as fh:
        for _ in range(10):
            fh.write(json.dumps(sample) + "\n")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_map, le = load_dataset(str(data_file))
    assert X_train.shape[1] == len(feature_map['columns'])
    assert len(le.classes_) == 1
