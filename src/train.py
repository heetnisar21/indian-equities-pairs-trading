"""Leakage-aware cross-sectional daily-return benchmark on NIFTY-50 data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FEATURES = ["ret_1", "ret_5", "ret_20", "vol_20", "range", "volume_z", "sma_gap"]


def load_panel() -> pd.DataFrame:
    frames = []
    for path in RAW.glob("*.csv"):
        if path.name == "NIFTY50_all.csv":
            continue
        try:
            x = pd.read_csv(path)
            x.columns = [c.strip().lower() for c in x.columns]
            date = next((c for c in x if c in {"date", "timestamp"}), None)
            close = next((c for c in x if c in {"close", "adj close", "adj_close"}), None)
            if not date or not close or len(x) < 300:
                continue
            x = x.rename(columns={date: "date", close: "close"})
            x["date"] = pd.to_datetime(x["date"], errors="coerce")
            x["close"] = pd.to_numeric(x["close"], errors="coerce")
            x["volume"] = pd.to_numeric(x.get("volume", np.nan), errors="coerce")
            x = x.dropna(subset=["date", "close"]).sort_values("date")
            x["symbol"] = path.stem
            frames.append(x[["date", "symbol", "close", "volume"]])
        except (ValueError, pd.errors.ParserError):
            continue
    if not frames:
        raise RuntimeError("No usable CSVs. Run scripts/download_kaggle.sh first.")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])


def engineer(panel: pd.DataFrame) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        r = g["close"].pct_change()
        g["ret_1"], g["ret_5"], g["ret_20"] = r, g["close"].pct_change(5), g["close"].pct_change(20)
        g["vol_20"] = r.rolling(20).std()
        g["range"] = (g["close"] / g["close"].rolling(20).max()) - 1
        lv = np.log1p(g["volume"])
        g["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
        g["sma_gap"] = g["close"] / g["close"].rolling(20).mean() - 1
        g["target"] = r.shift(-1)
        return g
    return panel.groupby("symbol", group_keys=False).apply(one, include_groups=False).dropna(subset=["target"])


def stats(y: np.ndarray, pred: np.ndarray) -> dict:
    # A deliberately simple, equal-weight daily long/short proxy with 5 bps turnover cost.
    signal = np.sign(pred)
    gross = signal * y
    turnover = np.abs(np.diff(signal, prepend=0))
    net = gross - 0.0005 * turnover
    return {
        "rmse_bps": round(float(np.sqrt(mean_squared_error(y, pred)) * 1e4), 3),
        "mae_bps": round(float(mean_absolute_error(y, pred) * 1e4), 3),
        "directional_accuracy": round(float(np.mean(np.sign(y) == np.sign(pred))), 4),
        "net_sharpe": round(float(np.sqrt(252) * net.mean() / net.std(ddof=1)), 3) if net.std() else 0.0,
        "mean_net_daily_bps": round(float(net.mean() * 1e4), 3),
    }


def main() -> None:
    df = engineer(load_panel()).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    # Remove malformed/split-like observations that overwhelm a daily-return model.
    df = df[df["target"].abs() <= 0.20].copy()
    df["ret_1"] = df["ret_1"].clip(-0.20, 0.20)
    df["ret_5"], df["ret_20"] = df["ret_5"].clip(-0.50, 0.50), df["ret_20"].clip(-1.0, 1.0)
    df["vol_20"], df["range"], df["volume_z"], df["sma_gap"] = (df["vol_20"].clip(0, 0.20), df["range"].clip(-1, 0), df["volume_z"].clip(-10, 10), df["sma_gap"].clip(-0.5, 0.5))
    cutoff = df["date"].quantile(0.80)
    train, test = df[df.date <= cutoff], df[df.date > cutoff]
    # A fixed-size sample keeps the tree benchmark fast on a laptop while preserving
    # the chronological holdout untouched.
    train_fit = train.sample(n=min(len(train), 60_000), random_state=7)
    X_train, y_train = train_fit[FEATURES], train_fit.target
    X_test, y_test = test[FEATURES], test.target.to_numpy()
    models = {
        "OLS": make_pipeline(SimpleImputer(), StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest": make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=100, min_samples_leaf=25, max_features=0.8, n_jobs=-1, random_state=7)),
        "Extra Trees": make_pipeline(SimpleImputer(), ExtraTreesRegressor(n_estimators=100, min_samples_leaf=25, max_features=0.8, n_jobs=-1, random_state=7)),
        "LightGBM": make_pipeline(SimpleImputer(), LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8, colsample_bytree=0.9, verbosity=-1, random_state=7)),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        row = {"model": name, **stats(y_test, model.predict(X_test))}
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False)
    REPORTS.mkdir(exist_ok=True)
    result.to_csv(REPORTS / "model_results.csv", index=False)
    meta = {"dataset": "Kaggle: rohanrao/nifty50-stock-market-data", "train_rows": len(train), "fit_rows": len(train_fit), "test_rows": len(test), "train_end": str(cutoff.date()), "features": FEATURES, "cost_assumption_bps_per_signal_change": 5}
    (REPORTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(result.to_string(index=False))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
