from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


# ===== SAME FEATURE SET =====
FEATURE_COLS = [
    "ret_1","ret_3","ret_5","ret_10",
    "vol_10","vol_20",
    "dist_sma_10","dist_sma_20",
    "zscore_20","atr_pct_14",
]

SELL_CLASS = -1
FLAT_CLASS = 0
BUY_CLASS = 1
CLASS_ORDER = [SELL_CLASS, FLAT_CLASS, BUY_CLASS]


# ================= DATA =================

def fetch_rates(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed")

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
    mt5.shutdown()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time","open","high","low","close","tick_volume"]].rename(columns={"tick_volume":"volume"})


def build_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy().sort_values("time")

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

    df["dist_sma_10"] = df["close"]/df["sma_10"] - 1
    df["dist_sma_20"] = df["close"]/df["sma_20"] - 1

    mean20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["zscore_20"] = (df["close"]-mean20)/std20

    tr = pd.concat([
        df["high"]-df["low"],
        (df["high"]-df["close"].shift()).abs(),
        (df["low"]-df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct_14"] = df["atr_14"]/df["close"]

    df["fwd_ret_h"] = df["close"].shift(-horizon)/df["close"] - 1

    return df.dropna()


def split_train_test(df: pd.DataFrame, train_ratio: float):
    split = int(len(df)*train_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def compute_return_barrier(df: pd.DataFrame, q: float):
    return float(df["fwd_ret_h"].abs().quantile(q))


def label_targets(df: pd.DataFrame, barrier: float):
    df = df.copy()
    df["target_class"] = 0
    df.loc[df["fwd_ret_h"] > barrier, "target_class"] = 1
    df.loc[df["fwd_ret_h"] < -barrier, "target_class"] = -1
    return df


# ================= MODEL =================

def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=25,
            weights="distance",
            metric="euclidean"
        ))
    ])


# ================= THRESHOLDS =================

def derive_thresholds(model, train_df, prob_q, gap_q):
    X = train_df[FEATURE_COLS].astype(np.float32)
    proba = model.predict_proba(X)

    p_sell, p_flat, p_buy = proba.T

    best = np.maximum(p_buy, p_sell)
    gap = best - np.maximum(p_flat, np.minimum(p_buy, p_sell))

    entry = float(np.quantile(best, prob_q))
    margin = float(np.quantile(gap, gap_q))

    return entry, margin


# ================= EXPORT =================

def export_onnx(model, path):
    initial_types = [("float_input", FloatTensorType([1,len(FEATURE_COLS)]))]
    onx = convert_sklearn(model, initial_types=initial_types, target_opset=15)
    path.write_bytes(onx.SerializeToString())


# ================= MAIN =================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAGUSD")
    p.add_argument("--timeframe", default="M15")
    p.add_argument("--bars", type=int, default=20000)
    p.add_argument("--horizon-bars", type=int, default=8)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--output-dir", default="output_knn")
    p.add_argument("--label-quantile", type=float, default=0.67)
    p.add_argument("--prob-quantile", type=float, default=0.8)
    p.add_argument("--margin-quantile", type=float, default=0.65)
    p.add_argument("--walk-forward-splits", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    df = fetch_rates(args.symbol, args.timeframe, args.bars)
    df = build_features(df, args.horizon_bars)

    train, test = split_train_test(df, args.train_ratio)

    barrier = compute_return_barrier(train, args.label_quantile)
    train = label_targets(train, barrier)
    test = label_targets(test, barrier)

    model = make_model()
    model.fit(train[FEATURE_COLS], train["target_class"])

    entry, gap = derive_thresholds(model, train, args.prob_quantile, args.margin_quantile)

    export_onnx(model, out/"ml_strategy_classifier_knn.onnx")

    meta = {
        "entry_prob_threshold": entry,
        "min_prob_gap": gap,
        "barrier": barrier
    }
    (out/"model_metadata.json").write_text(json.dumps(meta, indent=2))

    print("EntryProb:", entry)
    print("MinGap:", gap)


if __name__ == "__main__":
    main()
    