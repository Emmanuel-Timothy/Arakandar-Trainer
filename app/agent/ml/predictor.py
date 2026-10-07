"""
predictor.py
High-speed inference engine (Arakandar ML Engine - Phase 3).

Loads the champion model (or any saved checkpoint) and evaluates the
current market state in milliseconds. Returns calibrated class
probabilities plus the top driving features/tokens for explainability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from app.agent.ml.feature_pipeline import ALL_FEATURES, INV_LABEL_MAP
from app.agent.ml.trainer import load_model


class Predictor:
    def __init__(self, model_path: Path):
        self.model, self.scaler, self.features = load_model(model_path)

    def predict_proba(self, feature_row: pd.Series) -> Dict[str, float]:
        x = feature_row[self.features].astype(float).to_frame().T
        x_scaled = self.scaler.transform(x)
        proba = self.model.predict_proba(x_scaled)[0]
        return {INV_LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}

    def predict(self, feature_row: pd.Series) -> Dict:
        probs = self.predict_proba(feature_row)
        signal = max(probs, key=probs.get)
        confidence = probs[signal]
        drivers = self.top_drivers(feature_row, n=3)
        return {"signal": signal, "probabilities": probs, "confidence": confidence, "top_drivers": drivers}

    def top_drivers(self, feature_row: pd.Series, n: int = 3) -> List[str]:
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None and hasattr(self.model, "booster_"):
            importances = self.model.booster_.feature_importance(importance_type="gain")
        if importances is None:
            return []
        pairs = sorted(zip(self.features, importances), key=lambda p: p[1], reverse=True)[:n]
        return [name for name, _ in pairs]
