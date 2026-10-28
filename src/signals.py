"""Statistical Arbitrage Signal Engine & Dynamic Liquidity Gate."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional
import numpy as np
import pandas as pd

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from src.config import (
    CONTRACTS,
    DEFAULT_LOOKBACK_WINDOW,
    DEFAULT_Z_ENTRY,
    DEFAULT_Z_EXIT,
    DEFAULT_LIQUIDITY_PERCENTILE,
)

logger = logging.getLogger("bullionspread.signals")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def calculate_goldm_goldpetal_spread(
    df: pd.DataFrame,
    symbol_a: str = "GOLDM",
    symbol_b: str = "GOLDPETAL",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    data = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()

    # Calculate per-gram normalized price if not already cached
    if "norm_close" not in data.columns:
        data["norm_close"] = data["close"]
        for sym in (symbol_a, symbol_b):
            if sym in CONTRACTS:
                mask = data["symbol"] == sym
                mult = CONTRACTS[sym].per_gram_multiplier
                data.loc[mask, "norm_close"] = (data.loc[mask, "close"] * mult).round(3)

    df_a = data[data["symbol"] == symbol_a.strip().upper()].copy()
    df_b = data[data["symbol"] == symbol_b.strip().upper()].copy()

    if df_a.empty or df_b.empty:
        logger.warning("Required pair contracts missing from payload: %s, %s", symbol_a, symbol_b)
        return pd.DataFrame()

    aligned = pd.merge(
        df_a,
        df_b,
        on=["date", "expiry_date"],
        suffixes=(f"_{symbol_a.lower()}", f"_{symbol_b.lower()}"),
    )

    if aligned.empty:
        df_a_front = df_a.sort_values(["date", "expiry_date"]).groupby("date").first().reset_index()
        df_b_front = df_b.sort_values(["date", "expiry_date"]).groupby("date").first().reset_index()
        aligned = pd.merge(df_a_front, df_b_front, on="date", suffixes=(f"_{symbol_a.lower()}", f"_{symbol_b.lower()}"))
        if "expiry_date" not in aligned.columns and "expiry_date_x" in aligned.columns:
            aligned["expiry_date"] = aligned["expiry_date_x"]

    col_close_a = f"norm_close_{symbol_a.lower()}" if f"norm_close_{symbol_a.lower()}" in aligned.columns else "norm_close_x"
    col_close_b = f"norm_close_{symbol_b.lower()}" if f"norm_close_{symbol_b.lower()}" in aligned.columns else "norm_close_y"

    aligned["norm_close_a"] = aligned[col_close_a]
    aligned["norm_close_b"] = aligned[col_close_b]
    aligned["spread"] = (aligned["norm_close_a"] - aligned["norm_close_b"]).round(3)
    # Bug #3: Protect against ZeroDivisionError and Inf values when norm_close_b is 0 or NaN
    valid_close_b = aligned["norm_close_b"].where(aligned["norm_close_b"].abs() > 1e-6, np.nan)
    aligned["spread_pct"] = (
        (aligned["spread"] / valid_close_b) * 100.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0).round(4)

    aligned["volume_a"] = aligned.get(f"volume_{symbol_a.lower()}", aligned.get("volume_x", 0))
    aligned["volume_b"] = aligned.get(f"volume_{symbol_b.lower()}", aligned.get("volume_y", 0))
    aligned["open_interest_a"] = aligned.get(f"open_interest_{symbol_a.lower()}", aligned.get("open_interest_x", 0))
    aligned["open_interest_b"] = aligned.get(f"open_interest_{symbol_b.lower()}", aligned.get("open_interest_y", 0))
    aligned["symbol_a"] = symbol_a
    aligned["symbol_b"] = symbol_b

    keep_cols = [
        "date", "expiry_date", "symbol_a", "symbol_b",
        "norm_close_a", "norm_close_b", "spread", "spread_pct",
        "volume_a", "volume_b", "open_interest_a", "open_interest_b",
    ]
    return aligned[[c for c in keep_cols if c in aligned.columns]].sort_values("date").reset_index(drop=True)


def compute_spread_and_zscore(
    pair_df: pd.DataFrame,
    rolling_window: int = DEFAULT_LOOKBACK_WINDOW,
    min_periods: int = 5,
) -> pd.DataFrame:
    if pair_df.empty or "spread" not in pair_df.columns:
        return pair_df

    df = pair_df.copy().sort_values("date").reset_index(drop=True)

    # Missing session handling: forward-fill any spread voids prior to window calculation
    df["spread"] = df["spread"].ffill()

    df["rolling_mean"] = df["spread"].rolling(window=rolling_window, min_periods=min_periods).mean()
    df["rolling_std"] = df["spread"].rolling(window=rolling_window, min_periods=min_periods).std()

    # Bug #1: Numerical stability protection: zero or sub-epsilon std causes overflow or false signals
    valid_std = df["rolling_std"].where(df["rolling_std"].abs() > 1e-6, np.nan)
    z_raw = (df["spread"] - df["rolling_mean"]) / valid_std
    df["z_score"] = z_raw.replace([np.inf, -np.inf], 0.0).fillna(0.0).round(3)

    df["upper_band_2"] = (df["rolling_mean"] + 2.0 * df["rolling_std"]).round(3)
    df["lower_band_2"] = (df["rolling_mean"] - 2.0 * df["rolling_std"]).round(3)
    df["upper_band_1"] = (df["rolling_mean"] + 1.0 * df["rolling_std"]).round(3)
    df["lower_band_1"] = (df["rolling_mean"] - 1.0 * df["rolling_std"]).round(3)

    return df


def apply_liquidity_gates(
    df: pd.DataFrame,
    min_volume: Optional[float] = None,
    min_open_interest: Optional[float] = None,
    volume_percentile: Optional[float] = DEFAULT_LIQUIDITY_PERCENTILE,
    oi_percentile: Optional[float] = DEFAULT_LIQUIDITY_PERCENTILE,
    expanding_window: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    # Defensive liquidity flooring: User absolute floor overrides expanding percentile
    if min_volume is not None:
        out["vol_thresh_a"] = float(min_volume)
        out["vol_thresh_b"] = float(min_volume)
    else:
        pct = volume_percentile if volume_percentile is not None else 0.20
        if expanding_window and len(out) >= 10:
            out["vol_thresh_a"] = out["volume_a"].expanding(min_periods=10).quantile(pct).clip(lower=25)
            out["vol_thresh_b"] = out["volume_b"].expanding(min_periods=10).quantile(pct).clip(lower=25)
        else:
            out["vol_thresh_a"] = float(out["volume_a"].quantile(pct))
            out["vol_thresh_b"] = float(out["volume_b"].quantile(pct))

    if min_open_interest is not None:
        out["oi_thresh_a"] = float(min_open_interest)
        out["oi_thresh_b"] = float(min_open_interest)
    else:
        pct = oi_percentile if oi_percentile is not None else 0.20
        if expanding_window and len(out) >= 10:
            out["oi_thresh_a"] = out["open_interest_a"].expanding(min_periods=10).quantile(pct).clip(lower=100)
            out["oi_thresh_b"] = out["open_interest_b"].expanding(min_periods=10).quantile(pct).clip(lower=100)
        else:
            out["oi_thresh_a"] = float(out["open_interest_a"].quantile(pct))
            out["oi_thresh_b"] = float(out["open_interest_b"].quantile(pct))

    pass_vol = (out["volume_a"] >= out["vol_thresh_a"]) & (out["volume_b"] >= out["vol_thresh_b"])
    pass_oi = (out["open_interest_a"] >= out["oi_thresh_a"]) & (out["open_interest_b"] >= out["oi_thresh_b"])
    out["liquidity_pass"] = pass_vol & pass_oi

    def diagnose_gate(row):
        reasons = []
        if not (row["volume_a"] >= row["vol_thresh_a"]):
            reasons.append("LOW_VOL_A")
        if not (row["volume_b"] >= row["vol_thresh_b"]):
            reasons.append("LOW_VOL_B")
        if not (row["open_interest_a"] >= row["oi_thresh_a"]):
            reasons.append("LOW_OI_A")
        if not (row["open_interest_b"] >= row["oi_thresh_b"]):
            reasons.append("LOW_OI_B")
        return ", ".join(reasons) if reasons else "PASS"

    out["liquidity_status"] = out.apply(diagnose_gate, axis=1)
    return out


def generate_spread_signals(
    df: pd.DataFrame,
    z_entry_threshold: float = DEFAULT_Z_ENTRY,
    z_exit_threshold: float = DEFAULT_Z_EXIT,
) -> pd.DataFrame:
    if df.empty or "z_score" not in df.columns:
        return df

    out = df.copy()

    # Raw signal vector
    out["raw_signal"] = 0
    out.loc[out["z_score"] >= z_entry_threshold, "raw_signal"] = -1  # Overvalued spread (rich)
    out.loc[out["z_score"] <= -z_entry_threshold, "raw_signal"] = 1   # Undervalued spread (cheap)

    # Actionable signal vector: suppress whenever liquidity floor fails
    out["actionable_signal"] = out["raw_signal"]
    if "liquidity_pass" in out.columns:
        out.loc[~out["liquidity_pass"], "actionable_signal"] = 0

    def classify_regime(row):
        z = row["z_score"]
        liq = row.get("liquidity_pass", True)
        if abs(z) >= z_entry_threshold:
            if not liq:
                return "THIN_LIQUIDITY_SUPPRESSED"
            return "ANOMALY_SELL_SPREAD" if z > 0 else "ANOMALY_BUY_SPREAD"
        if abs(z) >= 1.0:
            return "WATCHLIST"
        return "FAIR_VALUE"

    out["market_regime"] = out.apply(classify_regime, axis=1)
    return out


def run_signal_pipeline(
    pair_or_bhavcopy_df: pd.DataFrame,
    symbol_a: str = "GOLDM",
    symbol_b: str = "GOLDPETAL",
    rolling_window: int = DEFAULT_LOOKBACK_WINDOW,
    min_volume: Optional[float] = None,
    min_open_interest: Optional[float] = None,
    volume_percentile: Optional[float] = DEFAULT_LIQUIDITY_PERCENTILE,
    oi_percentile: Optional[float] = DEFAULT_LIQUIDITY_PERCENTILE,
    z_entry_threshold: float = DEFAULT_Z_ENTRY,
    z_exit_threshold: float = DEFAULT_Z_EXIT,
) -> pd.DataFrame:
    if "spread" not in pair_or_bhavcopy_df.columns:
        pair_df = calculate_goldm_goldpetal_spread(pair_or_bhavcopy_df, symbol_a=symbol_a, symbol_b=symbol_b)
    else:
        pair_df = pair_or_bhavcopy_df

    df_z = compute_spread_and_zscore(pair_df, rolling_window=rolling_window)
    df_liq = apply_liquidity_gates(
        df_z,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        volume_percentile=volume_percentile,
        oi_percentile=oi_percentile,
    )
    return generate_spread_signals(df_liq, z_entry_threshold=z_entry_threshold, z_exit_threshold=z_exit_threshold)


def main():
    parser = argparse.ArgumentParser(description="MCX Bullion RV Signal Engine")
    parser.add_argument("--input", type=str, default="data/cleaned_bhavcopy.csv")
    parser.add_argument("--output", type=str, default="data/signals_goldm_goldpetal.csv")
    parser.add_argument("--window", type=int, default=DEFAULT_LOOKBACK_WINDOW)
    parser.add_argument("--z-entry", type=float, default=DEFAULT_Z_ENTRY)
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--min-oi", type=float, default=None)
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        alt = os.path.join(os.path.dirname(__file__), "..", input_path)
        if os.path.exists(alt):
            input_path = alt

    # Bug #16: Defensive CSV reading with error handling
    try:
        df_raw = pd.read_csv(input_path)
    except Exception as exc:
        logger.error("Failed to read Bhavcopy CSV input from %s: %s", input_path, exc)
        return

    df_signals = run_signal_pipeline(
        df_raw,
        symbol_a="GOLDM",
        symbol_b="GOLDPETAL",
        rolling_window=args.window,
        min_volume=args.min_volume,
        min_open_interest=args.min_oi,
        z_entry_threshold=args.z_entry,
    )

    # Bug #12: Guard logger against IndexError when DataFrame is empty or missing columns
    if not df_signals.empty and "spread" in df_signals.columns and "z_score" in df_signals.columns:
        logger.info(
            "Processed %d sessions. Latest spread: %+.2f INR/g (Z: %+.2f sigma)",
            len(df_signals),
            df_signals["spread"].iloc[-1],
            df_signals["z_score"].iloc[-1],
        )
        if "market_regime" in df_signals.columns:
            logger.info("Market regime breakdown: %s", df_signals["market_regime"].value_counts().to_dict())
    else:
        logger.warning("Processed %d sessions. DataFrame is empty or missing signal columns.", len(df_signals))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_signals.to_csv(args.output, index=False)
    logger.info("Persisted signals to %s", args.output)


if __name__ == "__main__":
    main()
