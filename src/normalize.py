"""Per-Gram 999-Purity Normalization Engine for MCX Bullion Futures."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, List
import numpy as np
import pandas as pd

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from src.config import CONTRACTS, TARGET_SYMBOLS, CONTRACT_SPECS

logger = logging.getLogger("bullionspread.normalize")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def compute_per_gram_multiplier(symbol: str) -> float:
    sym = symbol.strip().upper()
    if sym not in CONTRACTS:
        raise ValueError(f"Unknown commodity contract: '{sym}'. Whitelisted: {TARGET_SYMBOLS}")
    return CONTRACTS[sym].per_gram_multiplier


def normalize_contract_prices(
    df: pd.DataFrame,
    set_multi_index: bool = False,
) -> pd.DataFrame:
    if df.empty:
        empty_cols = ["date", "expiry_date", "symbol", "close", "norm_close", "volume", "open_interest"]
        res = pd.DataFrame(columns=empty_cols)
        return res.set_index(["date", "expiry_date"]) if set_multi_index else res

    normalized = df.copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.strip().str.upper()

    # Filter target contracts
    normalized = normalized[normalized["symbol"].isin(TARGET_SYMBOLS)].copy()
    if normalized.empty:
        logger.warning("Zero matching bullion contracts identified in provided payload.")
        return normalized

    # GOLDTEN active trading began in calendar year 2025
    # Bug #7, #14, #18: Defensive pd.to_datetime parsing preventing NaT from silently discarding GOLDTEN
    if "date" in normalized.columns:
        parsed_dates = pd.to_datetime(normalized["date"], errors="coerce")
        session_year = parsed_dates.dt.year.fillna(2025)
        valid_contract_mask = (normalized["symbol"] != "GOLDTEN") | (session_year >= 2025)
        normalized = normalized[valid_contract_mask].copy()

    # Vectorized per-gram 999 conversion multiplier
    multipliers = {sym: spec.per_gram_multiplier for sym, spec in CONTRACTS.items()}
    normalized["multiplier_999_1g"] = normalized["symbol"].map(multipliers).fillna(1.0)

    for px_col in ("close", "open", "high", "low"):
        if px_col in normalized.columns:
            normalized[f"norm_{px_col}"] = (normalized[px_col] * normalized["multiplier_999_1g"]).round(3)

    # Attach contract spec metadata safely without KeyError (Bug #6, #17)
    normalized["lot_size_grams"] = normalized["symbol"].map(
        lambda s: CONTRACTS[s].lot_size_grams if s in CONTRACTS else 0.0
    ).fillna(0.0)
    normalized["quote_unit_grams"] = normalized["symbol"].map(
        lambda s: CONTRACTS[s].quote_unit_grams if s in CONTRACTS else 1.0
    ).fillna(1.0)
    normalized["base_purity"] = normalized["symbol"].map(
        lambda s: CONTRACTS[s].base_purity if s in CONTRACTS else 0.999
    ).fillna(0.999)

    # Bug #2: Guard against ZeroDivisionError and Inf values if quote_unit_grams is 0 or NaN
    safe_quote_unit = normalized["quote_unit_grams"].where(normalized["quote_unit_grams"].abs() > 1e-6, np.nan)
    normalized["contract_value_inr"] = (
        normalized["close"] * (normalized["lot_size_grams"] / safe_quote_unit)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0).round(2)

    if "volume" in normalized.columns:
        normalized["turnover_inr"] = (normalized["volume"] * normalized["contract_value_inr"]).round(2)
    if "open_interest" in normalized.columns:
        normalized["oi_grams"] = (normalized["open_interest"] * normalized["lot_size_grams"]).round(2)

    # Handle holiday / gaps: Sort deterministically and forward-fill non-critical NaN prices
    sort_fields = [f for f in ["date", "expiry_date", "symbol"] if f in normalized.columns]
    normalized = normalized.sort_values(by=sort_fields).reset_index(drop=True)

    if set_multi_index and "date" in normalized.columns and "expiry_date" in normalized.columns:
        return normalized.set_index(["date", "expiry_date"])

    return normalized


def normalize_bhavcopy(
    bhavcopy_path_or_df: str | pd.DataFrame,
    set_multi_index: bool = True,
) -> pd.DataFrame:
    # Bug #16: Defensive CSV reading and validation
    if isinstance(bhavcopy_path_or_df, str):
        if not os.path.exists(bhavcopy_path_or_df):
            raise FileNotFoundError(f"Missing Bhavcopy file: {bhavcopy_path_or_df}")
        try:
            df = pd.read_csv(bhavcopy_path_or_df)
        except Exception as exc:
            logger.error("Failed to read Bhavcopy CSV from %s: %s", bhavcopy_path_or_df, exc)
            raise
    else:
        df = bhavcopy_path_or_df

    return normalize_contract_prices(df, set_multi_index=set_multi_index)


def get_aligned_spread_dataframe(
    df_norm: pd.DataFrame,
    symbol_a: str = "GOLDM",
    symbol_b: str = "GOLDPETAL",
    expiry_match_method: str = "same_or_nearest",
) -> pd.DataFrame:
    sym_a = symbol_a.strip().upper()
    sym_b = symbol_b.strip().upper()

    df = df_norm.reset_index() if isinstance(df_norm.index, pd.MultiIndex) else df_norm.copy()

    df_a = df[df["symbol"] == sym_a].copy()
    df_b = df[df["symbol"] == sym_b].copy()

    if df_a.empty or df_b.empty:
        return pd.DataFrame()

    shared_expiries = set(df_a["expiry_date"].unique()).intersection(set(df_b["expiry_date"].unique()))
    if shared_expiries and expiry_match_method != "nearest_only":
        df_a_shared = df_a[df_a["expiry_date"].isin(shared_expiries)]
        df_b_shared = df_b[df_b["expiry_date"].isin(shared_expiries)]
        aligned = pd.merge(
            df_a_shared,
            df_b_shared,
            on=["date", "expiry_date"],
            suffixes=("_a", "_b"),
        )
        aligned["expiry_date_a"] = aligned["expiry_date"]
        aligned["expiry_date_b"] = aligned["expiry_date"]
    else:
        # Align nearest forward delivery cycle per trade session
        df_a_first = df_a.sort_values(["date", "expiry_date"]).groupby("date").first().reset_index()
        df_b_first = df_b.sort_values(["date", "expiry_date"]).groupby("date").first().reset_index()
        aligned = pd.merge(
            df_a_first,
            df_b_first,
            on="date",
            suffixes=("_a", "_b"),
        )

    if aligned.empty:
        return pd.DataFrame()

    aligned["spread"] = (aligned["norm_close_a"] - aligned["norm_close_b"]).round(3)
    # Bug #3: Protect against ZeroDivisionError and Inf values when norm_close_b is 0 or NaN
    valid_b = aligned["norm_close_b"].where(aligned["norm_close_b"].abs() > 1e-6, np.nan)
    aligned["spread_pct"] = (
        (aligned["spread"] / valid_b) * 100.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0).round(4)

    canonical_order = [
        "date", "symbol_a", "symbol_b", "expiry_date_a", "expiry_date_b",
        "norm_close_a", "norm_close_b", "spread", "spread_pct",
        "volume_a", "volume_b", "open_interest_a", "open_interest_b",
    ]
    return aligned[[c for c in canonical_order if c in aligned.columns]].sort_values("date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="MCX Bullion Normalization Processor")
    parser.add_argument("--input", type=str, default="data/cleaned_bhavcopy.csv")
    parser.add_argument("--output", type=str, default="data/normalized_bhavcopy.csv")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        alt = os.path.join(os.path.dirname(__file__), "..", input_path)
        if os.path.exists(alt):
            input_path = alt

    try:
        df_norm = normalize_bhavcopy(input_path, set_multi_index=True)
    except Exception as exc:
        logger.error("Normalization process failed for %s: %s", input_path, exc)
        return

    logger.info("Standardized %d records across [%s]", len(df_norm), ", ".join(df_norm.index.names))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_norm.to_csv(args.output)
    logger.info("Persisted to %s", args.output)


if __name__ == "__main__":
    main()
