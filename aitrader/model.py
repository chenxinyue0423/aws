"""学习「波动惯性」的分类器：GradientBoosting 输出未来上涨概率。"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features as feat
from .config import CONFIG, MODEL_DIR


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])


def train_one(df_ohlcv: pd.DataFrame, horizon: int = CONFIG.horizon) -> tuple[Pipeline, dict]:
    feats = feat.build_features(df_ohlcv)
    label = feat.make_label(feats, horizon=horizon)
    aligned = feats.iloc[: -horizon].copy()
    aligned["y"] = label.iloc[: -horizon]
    aligned = aligned.dropna()

    X = aligned[feat.FEATURE_COLUMNS].values
    y = aligned["y"].values
    if len(np.unique(y)) < 2:
        raise RuntimeError("标签全部相同，无法训练")

    # 时间序列：用前 80% 训练，后 20% 验证。
    cut = int(len(X) * 0.8)
    pipe = _build_pipeline()
    pipe.fit(X[:cut], y[:cut])
    pred = pipe.predict(X[cut:])
    proba = pipe.predict_proba(X[cut:])[:, 1]
    metrics = {
        "n_train": int(cut),
        "n_val": int(len(X) - cut),
        "accuracy": float(accuracy_score(y[cut:], pred)),
        "auc": float(roc_auc_score(y[cut:], proba)) if len(np.unique(y[cut:])) > 1 else float("nan"),
    }
    return pipe, metrics


def model_path(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker.upper()}.joblib"


def save(ticker: str, pipe: Pipeline) -> Path:
    path = model_path(ticker)
    joblib.dump(pipe, path)
    return path


def load(ticker: str) -> Pipeline:
    return joblib.load(model_path(ticker))


def predict_proba(pipe: Pipeline, feats: pd.DataFrame) -> float:
    """对最新一根 K 线给出上涨概率。"""
    row = feats[feat.FEATURE_COLUMNS].iloc[[-1]].values
    return float(pipe.predict_proba(row)[0, 1])
