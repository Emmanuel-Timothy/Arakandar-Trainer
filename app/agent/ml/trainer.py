"""
trainer.py
Gradient Boosting Trainer for the Arakandar Decision Engine (Phase 3).

Implements three training modes:
  1. train_from_scratch(): fit a brand-new model on a full historical dataset.
  2. continue_training(): warm-start an existing model with more boosting
     rounds on newly arrived data (incremental training), without discarding
     previously learned trees.
  3. fine_tune(): adapt an already-trained champion model to a new
     ticker / market regime using a lower learning rate and fewer rounds,
     to avoid catastrophic forgetting of the general pattern.

ML engine note:
  The original plan specifies LightGBM. This trainer is written against a
  thin `GBMBackend` wrapper so you can swap engines with a one-line change:
    - If `lightgbm` is installed, it is used automatically (recommended,
      matches the production plan, supports true incremental `init_model`
      warm starts).
    - Otherwise it falls back to scikit-learn's `HistGradientBoostingClassifier`
      (same histogram-based GBDT family, use `warm_start=True` for continued
      training). This fallback lets you train and test the full pipeline
      completely offline with zero extra dependencies.

Custom asymmetric financial objective:
  Standard log-loss treats all misclassifications equally. Real trading
  does not: a false BUY signal before a crash is much more costly than a
  false HOLD. `financial_pnl_score` implements the asymmetric scoring
  function from the plan and is used for model *selection* / early-stopping
  comparisons across candidate models (works with either backend, since
  LightGBM's custom-objective hooks differ by version).
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from app.agent.ml.feature_pipeline import ALL_FEATURES, LABEL_MAP, INV_LABEL_MAP

try:
    import lightgbm as lgb
    BACKEND = "lightgbm"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    BACKEND = "sklearn_hgb"

MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHAMPION_PATH = MODEL_DIR / "champion_lgbm.pkl"


@dataclass
class TrainingReport:
    backend: str
    mode: str
    n_train: int
    n_val: int
    train_accuracy: float
    val_accuracy: float
    val_f1_macro: float
    val_log_loss: float
    net_pnl_score: float
    trained_at: str
    rounds_added: int
    feature_importance: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def financial_pnl_score(y_true: np.ndarray, y_pred: np.ndarray, fwd_returns: np.ndarray,
                          fee: float = 0.0015, penalty_lambda: float = 2.0) -> float:
    """Asymmetric Financial P&L scoring function (see plan formula):
      +  (Return - Fee)                      for a correct directional trade
      - (|Drawdown| * lambda + Fee)          for a wrong directional trade
      -  Slip                                for HOLD during a flat regime that moved
    y_pred / y_true are integer-encoded (0=SELL, 1=HOLD, 2=BUY).
    """
    score = 0.0
    for yt, yp, r in zip(y_true, y_pred, fwd_returns):
        is_long_call = yp == 2
        is_short_call = yp == 0
        is_hold_call = yp == 1
        if is_long_call:
            score += (r - fee) if r > 0 else -(abs(r) * penalty_lambda + fee)
        elif is_short_call:
            score += (-r - fee) if r < 0 else -(abs(r) * penalty_lambda + fee)
        elif is_hold_call:
            score += -abs(r) * 0.1
    return float(score / max(len(y_true), 1))


def _make_backend_model(n_estimators: int, learning_rate: float, max_depth: int,
                          init_model=None, warm_start: bool = False):
    if BACKEND == "lightgbm":
        params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective="multiclass",
            num_class=3,
            verbosity=-1,
            reg_alpha=0.5,
            reg_lambda=0.5,
            min_child_samples=20,
        )
        return lgb.LGBMClassifier(**params)
    else:
        return HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            warm_start=warm_start,
            l2_regularization=0.5,
            min_samples_leaf=20,
            random_state=42,
        )


def _class_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    balanced = {int(cls): len(y) / (len(classes) * count) for cls, count in zip(classes, counts)}
    # Square-root balancing improves minority recall without overwhelming the
    # probability calibration and directional precision of the base model.
    weights = {cls: float(weight) ** 0.5 for cls, weight in balanced.items()}
    return np.asarray([weights[int(value)] for value in y], dtype=float)


def _fit(model, X, y, init_model=None, eval_set=None):
    sample_weight = _class_weights(y)
    if BACKEND == "lightgbm":
        fit_params = {}
        if init_model is not None:
            fit_params["init_model"] = init_model
        if eval_set is not None:
            callbacks = [lgb.early_stopping(stopping_rounds=15, verbose=False)]
            model.fit(
                X, y, sample_weight=sample_weight,
                eval_set=eval_set, callbacks=callbacks, **fit_params,
            )
        else:
            model.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def _feature_importance(model, feature_names: List[str], X_val=None, y_val=None) -> dict:
    if BACKEND == "lightgbm":
        importances = model.booster_.feature_importance(importance_type="gain")
    else:
        importances = getattr(model, "feature_importances_", None)
        if importances is None and X_val is not None and y_val is not None:
            from sklearn.inspection import permutation_importance
            try:
                result = permutation_importance(
                    model, X_val, y_val, n_repeats=5, random_state=42, scoring="accuracy"
                )
                importances = np.clip(result.importances_mean, 0, None)
            except Exception:
                return {}
        if importances is None:
            return {}
    total = sum(importances) or 1.0
    pairs = sorted(zip(feature_names, importances), key=lambda p: p[1], reverse=True)
    return {name: round(float(val) / total, 4) for name, val in pairs[:15] if val > 0}


def _prepare_xy(X: pd.DataFrame, y: pd.Series, fwd: pd.Series, scaler: Optional[StandardScaler] = None):
    X = X[ALL_FEATURES].astype(float)
    y_enc = y.map(LABEL_MAP).astype(int)
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, y_enc.to_numpy(), fwd.to_numpy(), scaler


def _temporal_split(
    X: pd.DataFrame, y: pd.Series, fwd: pd.Series, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    split_at = int(len(X) * (1.0 - test_size))
    if split_at < 2 or split_at >= len(X):
        raise ValueError("not enough rows for the requested temporal split")
    return (
        X.iloc[:split_at], X.iloc[split_at:],
        y.iloc[:split_at], y.iloc[split_at:],
        fwd.iloc[:split_at], fwd.iloc[split_at:],
    )


def train_from_scratch(
    X: pd.DataFrame, y: pd.Series, fwd: pd.Series,
    n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 6,
    test_size: float = 0.2, save_path: Optional[Path] = None,
) -> Tuple[object, StandardScaler, TrainingReport]:
    """Fit a brand-new champion model from zero on the full dataset."""
    Xtr_raw, Xval_raw, ytr_raw, yval_raw, ftr, fval = _temporal_split(X, y, fwd, test_size)
    Xtr, ytr, _, scaler = _prepare_xy(Xtr_raw, ytr_raw, ftr)
    Xval, yval, _, _ = _prepare_xy(Xval_raw, yval_raw, fval, scaler=scaler)
    model = _make_backend_model(n_estimators, learning_rate, max_depth)
    _fit(model, Xtr, ytr, eval_set=[(Xval, yval)])

    report = _build_report(model, Xtr, ytr, Xval, yval, fval, "train_from_scratch", n_estimators)
    if save_path:
        save_model(model, scaler, save_path, report)
    return model, scaler, report


def walk_forward_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    fwd: pd.Series,
    horizon: int = 5,
    folds: int = 3,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 6,
) -> List[dict]:
    """Evaluate sequential folds without future leakage.

    Each validation window is preceded by a purge gap of ``horizon`` rows so
    training labels whose forward-return window reaches into validation data
    are excluded. A new scaler is fitted inside every fold.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if folds < 1:
        raise ValueError("folds must be at least one")
    if len(X) != len(y) or len(y) != len(fwd):
        raise ValueError("X, y, and fwd must have equal lengths")

    total = len(X)
    validation_size = total // (folds + 1)
    if validation_size < 1:
        raise ValueError("not enough rows for requested folds")
    results = []
    for fold in range(folds):
        validation_start = total - (folds - fold) * validation_size
        validation_end = validation_start + validation_size
        train_end = validation_start - horizon
        if train_end < 2:
            continue
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[validation_start:validation_end]
        y_val = y.iloc[validation_start:validation_end]
        fwd_val = fwd.iloc[validation_start:validation_end].to_numpy()
        if y_train.nunique() < 2 or y_val.empty:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[ALL_FEATURES].astype(float))
        X_val_scaled = scaler.transform(X_val[ALL_FEATURES].astype(float))
        model = _make_backend_model(n_estimators, learning_rate, max_depth)
        _fit(model, X_train_scaled, y_train.map(LABEL_MAP).astype(int).to_numpy())
        y_val_encoded = y_val.map(LABEL_MAP).astype(int).to_numpy()
        predictions = model.predict(X_val_scaled)
        probabilities = model.predict_proba(X_val_scaled)
        results.append({
            "fold": fold + 1,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "accuracy": round(float(accuracy_score(y_val_encoded, predictions)), 4),
            "f1_macro": round(float(f1_score(y_val_encoded, predictions, average="macro", zero_division=0)), 4),
            "log_loss": round(float(log_loss(y_val_encoded, probabilities, labels=[0, 1, 2])), 4),
            "net_pnl_score": round(financial_pnl_score(y_val_encoded, predictions, fwd_val), 6),
        })
    return results


def continue_training(
    existing_model_path: Path, X_new: pd.DataFrame, y_new: pd.Series, fwd_new: pd.Series,
    additional_rounds: int = 100, learning_rate: Optional[float] = None,
    test_size: float = 0.2, save_path: Optional[Path] = None,
) -> Tuple[object, StandardScaler, TrainingReport]:
    """Incrementally extend an existing model with new boosting rounds on
    freshly arrived market data, without discarding previously learned trees.
    """
    old_model, scaler, _ = load_model(existing_model_path)
    Xtr_raw, Xval_raw, ytr_raw, yval_raw, ftr, fval = _temporal_split(
        X_new, y_new, fwd_new, test_size
    )
    Xtr, ytr, _, scaler = _prepare_xy(Xtr_raw, ytr_raw, ftr, scaler=scaler)
    Xval, yval, _, _ = _prepare_xy(Xval_raw, yval_raw, fval, scaler=scaler)

    if BACKEND == "lightgbm":
        lr = learning_rate or old_model.get_params().get("learning_rate", 0.05)
        new_model = _make_backend_model(additional_rounds, lr, old_model.get_params()["max_depth"])
        _fit(new_model, Xtr, ytr, init_model=old_model.booster_)
    else:
        old_model.max_iter += additional_rounds
        old_model.warm_start = True
        _fit(old_model, Xtr, ytr)
        new_model = old_model

    report = _build_report(new_model, Xtr, ytr, Xval, yval, fval, "continue_training", additional_rounds)
    if save_path:
        save_model(new_model, scaler, save_path, report)
    return new_model, scaler, report


def fine_tune(
    champion_model_path: Path, X_target: pd.DataFrame, y_target: pd.Series, fwd_target: pd.Series,
    fine_tune_rounds: int = 50, fine_tune_lr: float = 0.01,
    test_size: float = 0.2, save_path: Optional[Path] = None,
) -> Tuple[object, StandardScaler, TrainingReport]:
    """Adapt a general champion model to a specific ticker / new regime
    using a small number of rounds and a low learning rate, minimizing
    catastrophic forgetting of previously learned general patterns.
    """
    champion_model, scaler, _ = load_model(champion_model_path)
    Xtr_raw, Xval_raw, ytr_raw, yval_raw, ftr, fval = _temporal_split(
        X_target, y_target, fwd_target, test_size
    )
    Xtr, ytr, _, scaler = _prepare_xy(Xtr_raw, ytr_raw, ftr, scaler=scaler)
    Xval, yval, _, _ = _prepare_xy(Xval_raw, yval_raw, fval, scaler=scaler)

    if BACKEND == "lightgbm":
        tuned_model = _make_backend_model(fine_tune_rounds, fine_tune_lr, champion_model.get_params()["max_depth"])
        _fit(tuned_model, Xtr, ytr, init_model=champion_model.booster_)
    else:
        champion_model.max_iter += fine_tune_rounds
        champion_model.learning_rate = fine_tune_lr
        champion_model.warm_start = True
        _fit(champion_model, Xtr, ytr)
        tuned_model = champion_model

    report = _build_report(tuned_model, Xtr, ytr, Xval, yval, fval, "fine_tune", fine_tune_rounds)
    if save_path:
        save_model(tuned_model, scaler, save_path, report)
    return tuned_model, scaler, report


def _build_report(model, Xtr, ytr, Xval, yval, fval, mode: str, rounds: int) -> TrainingReport:
    train_pred = model.predict(Xtr)
    val_pred = model.predict(Xval)
    val_proba = model.predict_proba(Xval)

    report = TrainingReport(
        backend=BACKEND,
        mode=mode,
        n_train=len(ytr),
        n_val=len(yval),
        train_accuracy=round(float(accuracy_score(ytr, train_pred)), 4),
        val_accuracy=round(float(accuracy_score(yval, val_pred)), 4),
        val_f1_macro=round(float(f1_score(yval, val_pred, average="macro")), 4),
        val_log_loss=round(float(log_loss(yval, val_proba, labels=[0, 1, 2])), 4),
        net_pnl_score=round(financial_pnl_score(yval, val_pred, fval), 6),
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        rounds_added=rounds,
        feature_importance=_feature_importance(model, ALL_FEATURES, Xval, yval),
    )
    return report


def save_model(model, scaler: StandardScaler, path: Path, report: TrainingReport):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "backend": BACKEND, "features": ALL_FEATURES}, f)
    report_path = path.with_suffix(".report.json")
    with open(report_path, "w") as f:
        f.write(report.to_json())


def load_model(path: Path):
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["scaler"], payload.get("features", ALL_FEATURES)


if __name__ == "__main__":
    print(f"Arakandar trainer ready. Backend detected: {BACKEND}")
