"""Regression checks: run after `python run_pipeline.py`. Fails CI if results drift outside expected ranges."""
import json
from pathlib import Path

import pytest

METRICS = json.loads((Path(__file__).resolve().parents[1] / 'reports' / 'metrics.json').read_text())


def get(path):
    v = METRICS
    for k in path.split('.') if path not in METRICS else [path]:
        v = v[k]
    return v


@pytest.mark.parametrize('path,op,expected', [
    ('roc_auc_oot', '>', 0.75),
    ('cox_c_index_oot', '>', 0.7),
    ('n_final_features', '>=', 8),
])
def test_metric(path, op, expected):
    v = get(path)
    ok = {'>': v > expected, '>=': v >= expected, '==': v == expected}[op]
    assert ok, f'{path} = {v}, expected {op} {expected}'
