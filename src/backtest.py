"""Chronological Walk-Forward Backtester & Statutory Cost Attribution Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

import os
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from src.config import (
    DEFAULT_FEES,
    DEFAULT_EXPIRY_GUARD_DAYS,
    DEFAULT_Z_ENTRY,
    DEFAULT_Z_EXIT,
    DEFAULT_STOP_LOSS_Z,
    StatutoryFees,
)


@dataclass
class TradeRecord:
    trade_id: int
    pair: str
    direction: str
    entry_date: str
    exit_date: str
    entry_spread: float
    exit_spread: float
    entry_z: float
    exit_z: float
    holding_days: int
    gross_pnl: float
    transaction_costs: float
    net_pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    metrics: Dict[str, Any]
    daily_curve: pd.DataFrame
    trades_df: pd.DataFrame
    cost_attribution: Dict[str, float]


def run_walk_forward_backtest(
    signal_df: pd.DataFrame,
    z_entry_threshold: float = DEFAULT_Z_ENTRY,
    z_exit_threshold: float = DEFAULT_Z_EXIT,
    days_before_expiry_exit: int = DEFAULT_EXPIRY_GUARD_DAYS,
    slippage_bps_per_leg: float = 2.5,
    stop_loss_z: float = DEFAULT_STOP_LOSS_Z,
    margin_requirement_pct: float = 0.15,
    risk_free_rate_annual: float = 0.065,
    fee_model: Optional[StatutoryFees] = None,
) -> BacktestResult:
    fees = fee_model or DEFAULT_FEES

    if signal_df.empty or "spread" not in signal_df.columns:
        empty_metrics = {
            "total_net_pnl": 0.0,
            "total_gross_pnl": 0.0,
            "total_costs": 0.0,
            "net_return_on_margin_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_inr": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "avg_trade_pnl": 0.0,
            "avg_holding_days": 0.0,
            "gold_drift_inr": 0.0,
            "strategy_beta": 0.0,
            "allocated_margin_inr": 0.0,
        }
        return BacktestResult(
            metrics=empty_metrics,
            daily_curve=pd.DataFrame(),
            trades_df=pd.DataFrame(),
            cost_attribution={},
        )

    df = signal_df.copy().sort_values("date").reset_index(drop=True)

    # Compute days to tender period (Bug #14: defensive pd.to_datetime parsing)
    exp_col = "expiry_date_a" if "expiry_date_a" in df.columns else ("expiry_date" if "expiry_date" in df.columns else None)
    if exp_col:
        exp_dates = pd.to_datetime(df[exp_col], errors="coerce")
        trade_dates = pd.to_datetime(df["date"], errors="coerce")
        df["days_to_expiry"] = (exp_dates - trade_dates).dt.days.fillna(999).astype(int)
    else:
        df["days_to_expiry"] = 999

    n = len(df)
    positions = np.zeros(n, dtype=int)
    daily_gross_pnl = np.zeros(n, dtype=float)
    daily_costs = np.zeros(n, dtype=float)
    trade_actions = ["HOLD"] * n

    trades: List[TradeRecord] = []
    trade_id_counter = 1

    current_pos = 0
    entry_idx = 0
    entry_spread = 0.0
    entry_z = 0.0

    cost_accumulator = {
        "mcx_turnover_fee": 0.0,
        "sebi_turnover_fee": 0.0,
        "ctt": 0.0,
        "stamp_duty": 0.0,
        "brokerage": 0.0,
        "gst": 0.0,
        "estimated_slippage": 0.0,
    }

    for i in range(n):
        row = df.iloc[i]
        curr_spread = row["spread"]
        curr_z = row.get("z_score", 0.0)
        curr_liq = row.get("liquidity_pass", True)
        curr_dte = row.get("days_to_expiry", 999)
        px_a = row.get("norm_close_a", row.get("close", 7500.0))
        px_b = row.get("norm_close_b", 7500.0)

        # Mark-to-market spread PnL from T-1 position
        if i > 0 and current_pos != 0:
            spread_diff = curr_spread - df.iloc[i - 1]["spread"]
            daily_gross_pnl[i] += current_pos * spread_diff

        # Evaluate exits
        if current_pos != 0:
            exit_triggered = False
            exit_reason = ""

            # Tender period exit rule (mandatory liquidation 5 days prior to contract expiry)
            if curr_dte <= days_before_expiry_exit:
                exit_triggered = True
                exit_reason = "EXPIRY_TENDER_GUARD"
            elif current_pos == 1 and curr_z >= -z_exit_threshold:
                exit_triggered = True
                exit_reason = "MEAN_REVERSION"
            elif current_pos == -1 and curr_z <= z_exit_threshold:
                exit_triggered = True
                exit_reason = "MEAN_REVERSION"
            elif abs(curr_z) >= stop_loss_z:
                exit_triggered = True
                exit_reason = "STOP_LOSS_DIVERGENCE"

            if exit_triggered:
                # Calculate round-trip statutory costs on completion
                exit_breakdown = fees.calculate_round_trip_breakdown(
                    notional_leg_a=px_a,
                    notional_leg_b=px_b,
                    slippage_bps=slippage_bps_per_leg,
                )
                round_trip_friction = exit_breakdown["total_round_trip_cost"]
                daily_costs[i] += round_trip_friction

                for key in cost_accumulator:
                    cost_accumulator[key] += exit_breakdown[key]

                trade_gross = current_pos * (curr_spread - entry_spread)
                trade_net = trade_gross - round_trip_friction
                dir_label = "LONG_SPREAD" if current_pos == 1 else "SHORT_SPREAD"

                # Bug #8: Safe entry_date indexing
                entry_date_val = df.iloc[entry_idx]["date"] if (0 <= entry_idx < len(df)) else str(row.get("date", ""))

                trades.append(
                    TradeRecord(
                        trade_id=trade_id_counter,
                        pair=f"{row.get('symbol_a', 'GOLDM')}/{row.get('symbol_b', 'GOLDPETAL')}",
                        direction=dir_label,
                        entry_date=entry_date_val,
                        exit_date=row["date"],
                        entry_spread=round(entry_spread, 3),
                        exit_spread=round(curr_spread, 3),
                        entry_z=round(entry_z, 2),
                        exit_z=round(curr_z, 2),
                        holding_days=i - entry_idx,
                        gross_pnl=round(trade_gross, 3),
                        transaction_costs=round(round_trip_friction, 3),
                        net_pnl=round(trade_net, 3),
                        exit_reason=exit_reason,
                    )
                )
                trade_id_counter += 1
                trade_actions[i] = f"EXIT_{exit_reason}"
                current_pos = 0

        # Evaluate entries
        if current_pos == 0:
            eligible_for_entry = curr_liq and (curr_dte > days_before_expiry_exit)
            if eligible_for_entry:
                if curr_z <= -z_entry_threshold:
                    current_pos = 1
                    entry_idx = i
                    entry_spread = curr_spread
                    entry_z = curr_z
                    trade_actions[i] = "ENTER_LONG_SPREAD"
                elif curr_z >= z_entry_threshold:
                    current_pos = -1
                    entry_idx = i
                    entry_spread = curr_spread
                    entry_z = curr_z
                    trade_actions[i] = "ENTER_SHORT_SPREAD"

        positions[i] = current_pos

    df["position"] = positions
    df["daily_gross_pnl"] = daily_gross_pnl
    df["daily_costs"] = daily_costs
    df["daily_net_pnl"] = df["daily_gross_pnl"] - df["daily_costs"]
    df["cum_gross_pnl"] = df["daily_gross_pnl"].cumsum().round(3)
    df["cum_costs"] = df["daily_costs"].cumsum().round(3)
    df["cum_net_pnl"] = df["daily_net_pnl"].cumsum().round(3)
    df["trade_action"] = trade_actions

    # Raw gold drift benchmark (Bug #6: safe iloc[0] indexing)
    bench_col = "norm_close_a" if "norm_close_a" in df.columns else "norm_close"
    if bench_col in df.columns and len(df) > 0:
        df["gold_benchmark_price"] = df[bench_col]
        df["gold_daily_return"] = df["gold_benchmark_price"].pct_change().fillna(0.0)
        base_bench = df["gold_benchmark_price"].iloc[0]
        df["gold_market_drift_inr"] = (df["gold_benchmark_price"] - base_bench).round(3)
    else:
        df["gold_daily_return"] = 0.0
        df["gold_market_drift_inr"] = 0.0

    avg_price = df["gold_benchmark_price"].mean() if ("gold_benchmark_price" in df.columns and len(df) > 0) else 7500.0
    allocated_margin = avg_price * 2.0 * margin_requirement_pct
    df["strategy_daily_return"] = (df["daily_net_pnl"] / allocated_margin) if allocated_margin > 0 else 0.0

    # Market beta orthogonality check
    if len(df) > 1:
        cov_mat = np.cov(df["strategy_daily_return"], df["gold_daily_return"])
        var_mkt = cov_mat[1, 1]
        strategy_beta = (cov_mat[0, 1] / var_mkt) if var_mkt > 1e-12 else 0.0
    else:
        strategy_beta = 0.0

    # Drawdown profile
    running_peak = np.maximum.accumulate(df["cum_net_pnl"])
    drawdown_inr = df["cum_net_pnl"] - running_peak
    df["drawdown_inr"] = drawdown_inr.round(3)
    max_dd_inr = float(abs(drawdown_inr.min())) if len(drawdown_inr) > 0 else 0.0
    max_dd_pct = (max_dd_inr / allocated_margin * 100.0) if allocated_margin > 0 else 0.0

    # Risk-adjusted performance (Bug #5: robust Sortino calculation preventing NaN truth check)
    daily_rf = risk_free_rate_annual / 252.0
    excess_ret = df["strategy_daily_return"] - daily_rf
    std_ret = df["strategy_daily_return"].std()
    downside_std = df[df["strategy_daily_return"] < 0]["strategy_daily_return"].std()

    sharpe = float(np.sqrt(252) * (excess_ret.mean() / std_ret)) if (pd.notna(std_ret) and std_ret > 1e-8) else 0.0
    sortino = (
        float(np.sqrt(252) * (excess_ret.mean() / downside_std))
        if (pd.notna(downside_std) and not np.isnan(downside_std) and downside_std > 1e-8)
        else 0.0
    )

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    total_trades = len(trades_df)
    if total_trades > 0:
        winners = trades_df[trades_df["net_pnl"] > 0]
        losers = trades_df[trades_df["net_pnl"] < 0]
        win_rate = (len(winners) / total_trades) * 100.0
        gross_w = winners["net_pnl"].sum()
        gross_l = abs(losers["net_pnl"].sum())
        # Bug #19: When gross_l == 0 and gross_w == 0, return 0.0 instead of misleading 1.0
        profit_factor = (gross_w / gross_l) if gross_l > 0 else (99.0 if gross_w > 0 else 0.0)
        avg_trade = trades_df["net_pnl"].mean()
        avg_hold = trades_df["holding_days"].mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0
        avg_hold = 0.0

    # Bug #7: Guard iloc[-1] against empty dataframe
    total_net = float(df["cum_net_pnl"].iloc[-1]) if len(df) > 0 else 0.0
    total_gross = float(df["cum_gross_pnl"].iloc[-1]) if len(df) > 0 else 0.0
    total_costs = float(df["cum_costs"].iloc[-1]) if len(df) > 0 else 0.0
    gold_drift = float(df["gold_market_drift_inr"].iloc[-1]) if len(df) > 0 else 0.0
    net_return_on_margin = (total_net / allocated_margin * 100.0) if allocated_margin > 0 else 0.0

    metrics = {
        "total_net_pnl": round(float(total_net), 2),
        "total_gross_pnl": round(float(total_gross), 2),
        "total_costs": round(float(total_costs), 2),
        "net_return_on_margin_pct": round(float(net_return_on_margin), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "sortino_ratio": round(float(sortino), 2),
        "max_drawdown_inr": round(float(max_dd_inr), 2),
        "max_drawdown_pct": round(float(max_dd_pct), 2),
        "win_rate_pct": round(float(win_rate), 1),
        "profit_factor": round(float(profit_factor), 2),
        "total_trades": int(total_trades),
        "avg_trade_pnl": round(float(avg_trade), 2),
        "avg_holding_days": round(float(avg_hold), 1),
        "gold_drift_inr": round(float(gold_drift), 2),
        "strategy_beta": round(float(strategy_beta), 3),
        "allocated_margin_inr": round(float(allocated_margin), 2),
    }

    rounded_cost_breakdown = {k: round(v, 2) for k, v in cost_accumulator.items()}

    return BacktestResult(
        metrics=metrics,
        daily_curve=df,
        trades_df=trades_df,
        cost_attribution=rounded_cost_breakdown,
    )
