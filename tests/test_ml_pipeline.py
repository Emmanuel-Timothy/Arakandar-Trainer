"""tests/test_ml_pipeline.py — Feature pipeline + trainer sanity tests."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.agent.ml.feature_pipeline import build_feature_matrix, make_labels, ALL_FEATURES
from app.agent.ml.trainer import train_from_scratch, financial_pnl_score, walk_forward_evaluate


def _synthetic_df(n=300, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    ret = rng.normal(0.0003, 0.015, n)
    close = 1000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.lognormal(mean=15, sigma=0.3, size=n).astype(int)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_feature_matrix_shape_and_bounds():
    df = _synthetic_df()
    X, y, fwd, meta = build_feature_matrix(df)
    assert list(X.columns) == ALL_FEATURES
    assert len(X) == len(y) == len(fwd) == len(meta)
    assert set(y.unique()).issubset({"BUY", "HOLD", "SELL"})


def test_make_labels_uses_future_return_and_thresholds():
    df = pd.DataFrame({"close": [100.0, 101.9, 98.0, 100.0]})

    labels, fwd_return = make_labels(df, horizon=1, up_thresh=0.02, down_thresh=-0.02)

    assert np.allclose(fwd_return.iloc[:3], [0.019, -0.03827282, 0.02040816])
    assert labels.tolist() == ["HOLD", "SELL", "BUY", "HOLD"]


def test_training_and_inference_latency():
    df = _synthetic_df(n=400)
    X, y, fwd, meta = build_feature_matrix(df)
    model, scaler, report = train_from_scratch(X, y, fwd, n_estimators=50)
    assert report.val_accuracy >= 0.0
    x_sample = scaler.transform(X.iloc[[0]][ALL_FEATURES].astype(float))
    start = time.perf_counter()
    proba = model.predict_proba(x_sample)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert abs(sum(proba) - 1.0) < 1e-6
    assert elapsed_ms < 50


def test_financial_pnl_score_penalizes_wrong_calls():
    y_true = np.array([2, 0])
    y_pred_correct = np.array([2, 0])
    y_pred_wrong = np.array([0, 2])
    fwd = np.array([0.05, -0.05])
    correct_score = financial_pnl_score(y_true, y_pred_correct, fwd)
    wrong_score = financial_pnl_score(y_true, y_pred_wrong, fwd)
    assert correct_score > wrong_score


def test_walk_forward_evaluation_purges_horizon_gap():
    df = _synthetic_df(n=500)
    X, y, fwd, _ = build_feature_matrix(df)

    folds = walk_forward_evaluate(X, y, fwd, horizon=5, folds=2, n_estimators=10)

    assert len(folds) == 2
    for fold in folds:
        assert fold["train_end"] + 5 == fold["validation_start"]
        assert 0.0 <= fold["accuracy"] <= 1.0
