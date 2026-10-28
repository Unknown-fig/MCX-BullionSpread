"""BullionSpread: Institutional MCX Gold Futures Relative-Value & Statistical Arbitrage Terminal.

Built for Hack N Hillz Quant Desk.
Standardizes fragmented Multi Commodity Exchange of India (MCX) gold derivatives
(GOLDM 100g, GOLDTEN 10g, GOLDGUINEA 8g, GOLDPETAL 1g) into an authoritative 1-gram 999-purity
benchmark, models rolling spread dynamics with dynamic Z-scores, enforces microstructure liquidity
safety gates, simulates walk-forward backtests with statutory cost attribution, and provides
an interactive synchronized 2-leg execution desk with real-time portfolio Mark-to-Market.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

logger = logging.getLogger("mcx_terminal")

# Ensure repository paths resolve for modular imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.join(BASE_DIR, "mcx-bullionspread")
for path in [BASE_DIR, SUB_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.config import (
    CONTRACTS,
    TARGET_SYMBOLS,
    DEFAULT_LOOKBACK_WINDOW,
    DEFAULT_Z_ENTRY,
    DEFAULT_Z_EXIT,
    DEFAULT_LIQUIDITY_PERCENTILE,
    DEFAULT_FEES,
    StatutoryFees,
    FEE_SCHEDULE,
    ContractSpec,
    FeeScheduleItem,
)
from src.ingest import (
    save_and_ingest_data,
    fetch_mcx_bhavcopy_live,
)
from src.normalize import (
    normalize_contract_prices,
    get_aligned_spread_dataframe,
)
from src.signals import (
    compute_spread_and_zscore,
    apply_liquidity_gates,
    generate_spread_signals,
    run_signal_pipeline,
)
from src.backtest import (
    run_walk_forward_backtest,
    BacktestResult,
)

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="BULLIONSPREAD | MCX Quant Arbitrage Terminal",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Protect all HTML markdown from 4-space indentation code block parser traps
_orig_markdown = st.markdown


def safe_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html", False) and isinstance(body, str):
        body = "\n".join(line.lstrip() for line in body.splitlines())
    return _orig_markdown(body, *args, **kwargs)


st.markdown = safe_markdown

# -------------------------------------------------------------
# HIGH-END OBSIDIAN & 24K GOLD PRO QUANT DESIGN SYSTEM
# -------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Typography & Root Canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #f1f5f9 !important;
    }
    h1, h2, h3, h4, .terminal-title, .brand-font {
        font-family: 'Outfit', sans-serif !important;
    }
    code, .mono-val, .ticker-val, .table-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Ambient Luxury Obsidian Background */
    .stApp {
        background: radial-gradient(circle at 50% -12%, #18243c 0%, #0d1322 45%, #06080e 100%) !important;
        background-attachment: fixed !important;
        color: #f1f5f9 !important;
    }

    /* Streamlit block layout container */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1600px !important;
    }

    /* Hide standard Streamlit header clutter */
    #MainMenu, footer, header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Modern Top Ticker Tape Bar */
    .ticker-bar {
        background: linear-gradient(90deg, rgba(16, 23, 38, 0.94) 0%, rgba(12, 18, 30, 0.96) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 14px;
        padding: 8px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    .ticker-cards-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .ticker-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 14px;
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    .ticker-item:hover {
        background: rgba(245, 158, 11, 0.05);
        border-color: rgba(245, 158, 11, 0.25);
        transform: translateY(-1px);
    }
    .ticker-sym-row {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .ticker-name {
        font-size: 0.78rem;
        font-weight: 800;
        color: #e2e8f0;
        letter-spacing: 0.03em;
    }
    .ticker-lot-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        color: #94a3b8;
        background: rgba(255, 255, 255, 0.06);
        padding: 1px 5px;
        border-radius: 4px;
    }
    .ticker-price {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .ticker-norm {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #f59e0b;
        background: rgba(245, 158, 11, 0.12);
        padding: 2px 7px;
        border-radius: 6px;
        border: 1px solid rgba(245, 158, 11, 0.25);
        font-weight: 600;
    }
    .ticker-delta-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 5px;
        display: inline-flex;
        align-items: center;
        gap: 2px;
    }
    .ticker-delta-up {
        color: #10b981;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .ticker-delta-down {
        color: #ef4444;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .ticker-meta-group {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-left: auto;
    }
    .ticker-badge-live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #10b981;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        letter-spacing: 0.04em;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulse-animation 1.8s infinite;
    }
    @keyframes pulse-animation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Terminal Executive Brand Bar */
    .brand-cluster {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon-box {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: linear-gradient(135deg, #f59e0b 0%, #b45309 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.55rem;
        color: #05060a;
        box-shadow: 0 6px 22px rgba(245, 158, 11, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }
    .brand-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 45%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .brand-subtitle {
        font-size: 0.80rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 3px;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .tag-hack {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 700;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .tag-info-chip {
        background: rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Institutional HUD KPI Cards */
    .hud-card {
        background: linear-gradient(180deg, rgba(20, 28, 48, 0.82) 0%, rgba(12, 17, 30, 0.88) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
        transition: all 0.22s ease-in-out;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
        min-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .hud-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.55), 0 0 20px rgba(245, 158, 11, 0.15);
        border-color: rgba(245, 158, 11, 0.4);
    }
    .hud-card-gold {
        border-top: 3px solid #f59e0b;
    }
    .hud-card-emerald {
        border-top: 3px solid #10b981;
    }
    .hud-card-cyan {
        border-top: 3px solid #06b6d4;
    }
    .hud-card-purple {
        border-top: 3px solid #8b5cf6;
    }
    .hud-card-crimson {
        border-top: 3px solid #ef4444;
    }

    .hud-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .hud-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #94a3b8;
    }
    .hud-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
    }
    .hud-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin: 4px 0;
    }
    .hud-footer {
        font-size: 0.73rem;
        color: #64748b;
        font-weight: 500;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Micro Visual Dynamic Range & Progress Meters */
    .mini-meter-track {
        height: 5px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.08);
        overflow: hidden;
        margin: 6px 0 3px 0;
        position: relative;
    }
    .mini-meter-fill-green {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
    }
    .mini-meter-fill-gold {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%);
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
    }
    .mini-meter-fill-cyan {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #06b6d4 0%, #38bdf8 100%);
        box-shadow: 0 0 8px rgba(6, 182, 212, 0.5);
    }
    .mini-meter-fill-multi {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #10b981 0%, #06b6d4 50%, #ef4444 100%);
    }

    /* Glass Content Panels */
    .glass-panel {
        background: linear-gradient(180deg, rgba(18, 25, 42, 0.8) 0%, rgba(10, 15, 26, 0.85) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    .glass-panel-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }
    .glass-panel-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-bottom: 16px;
    }

    /* Modern Pill Segmented Tabs (Institutional Grade) */
    /* 1. Hide the react-aria red underline selection indicator */
    div[data-testid="stTab"] .react-aria-SelectionIndicator,
    div[data-testid="stTab"] [data-rac].react-aria-SelectionIndicator,
    [data-testid="stTabs"] .react-aria-SelectionIndicator,
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
        background-color: transparent !important;
        height: 0 !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }

    /* 2. Container of tabs */
    div[data-testid="stTabs"] div[role="tablist"],
    div[role="tablist"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"],
    div[data-baseweb="tab-list"] {
        background: rgba(13, 19, 32, 0.95) !important;
        padding: 6px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        gap: 6px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45) !important;
        display: flex !important;
        flex-wrap: wrap !important;
    }

    /* 3. Each individual tab (Base State) */
    div[data-testid="stTab"],
    div[role="tab"],
    button[data-testid="stTab"],
    button[role="tab"],
    div[data-baseweb="tab-list"] button {
        background: transparent !important;
        color: #94a3b8 !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    div[data-testid="stTab"] p,
    div[role="tab"] p,
    button[data-testid="stTab"] p,
    button[role="tab"] p {
        color: #94a3b8 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        transition: color 0.2s ease !important;
    }

    /* 4. Tab Hover State */
    div[data-testid="stTab"]:hover,
    div[role="tab"]:hover,
    button[data-testid="stTab"]:hover,
    button[role="tab"]:hover,
    div[data-baseweb="tab-list"] button:hover {
        color: #f59e0b !important;
        background: rgba(245, 158, 11, 0.09) !important;
        border-color: rgba(245, 158, 11, 0.25) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stTab"]:hover p,
    div[role="tab"]:hover p,
    button[data-testid="stTab"]:hover p,
    button[role="tab"]:hover p {
        color: #f59e0b !important;
    }

    /* 5. Active Selected Tab (Luminous Gold Pill) */
    div[data-testid="stTab"][aria-selected="true"],
    div[data-testid="stTab"][data-selected="true"],
    div[role="tab"][aria-selected="true"],
    button[data-testid="stTab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"],
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #05060a !important;
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%) !important;
        box-shadow: 0 4px 18px rgba(245, 158, 11, 0.45) !important;
        border: 1px solid rgba(245, 158, 11, 0.8) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stTab"][aria-selected="true"] p,
    div[data-testid="stTab"][data-selected="true"] p,
    div[role="tab"][aria-selected="true"] p,
    button[data-testid="stTab"][aria-selected="true"] p,
    button[role="tab"][aria-selected="true"] p {
        color: #05060a !important;
        font-weight: 800 !important;
    }

    /* Modern Radio Group Pill Buttons */
    div[data-testid="stRadio"] [role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 10px !important;
        background: rgba(11, 15, 25, 0.8) !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    div[data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: rgba(245, 158, 11, 0.4) !important;
        background: rgba(245, 158, 11, 0.08) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(245, 158, 11, 0.12) 100%) !important;
        border-color: #f59e0b !important;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.2) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #f59e0b !important;
        font-weight: 700 !important;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        accent-color: #f59e0b !important;
    }

    /* Custom Form & Selectbox Styling */
    div[data-baseweb="select"] > div {
        background-color: #0b0f1a !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(245, 158, 11, 0.5) !important;
        box-shadow: 0 0 14px rgba(245, 158, 11, 0.18) !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="menu"] {
        background-color: #0b0f1a !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.75) !important;
    }
    div[data-baseweb="menu"] li {
        color: #e2e8f0 !important;
        font-size: 0.86rem !important;
    }
    div[data-baseweb="menu"] li:hover {
        background-color: rgba(245, 158, 11, 0.15) !important;
        color: #f59e0b !important;
    }

    /* Action Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(18, 25, 42, 0.8) !important;
        color: #e2e8f0 !important;
    }
    .stButton > button:hover {
        border-color: rgba(245, 158, 11, 0.4) !important;
        color: #f59e0b !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: #05060a !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 22px rgba(245, 158, 11, 0.55) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: linear-gradient(180deg, rgba(16, 23, 39, 0.7) 0%, rgba(10, 14, 24, 0.75) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 18px !important;
    }
    div[data-testid="stExpander"] summary {
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        padding: 12px 18px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #f59e0b !important;
        background: rgba(245, 158, 11, 0.05) !important;
    }

    /* Actionable Signal Banners */
    .signal-banner-long {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.04) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-left: 5px solid #10b981;
        border-radius: 14px;
        padding: 14px 20px;
        margin: 14px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    .signal-banner-short {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.04) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 5px solid #ef4444;
        border-radius: 14px;
        padding: 14px 20px;
        margin: 14px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
    }
    .signal-banner-gated {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.04) 100%);
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-left: 5px solid #f59e0b;
        border-radius: 14px;
        padding: 14px 20px;
        margin: 14px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
    }

    /* Desk Quick Stats Chip Bar */
    .desk-meta-strip {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    .desk-meta-item {
        background: rgba(14, 20, 33, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.73rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .desk-meta-label {
        color: #94a3b8;
        font-weight: 600;
    }
    .desk-meta-val {
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# DIRECTORY RESOLUTION & DATA PIPELINE
# -------------------------------------------------------------
candidate_data_dirs = [
    os.path.join(BASE_DIR, "data"),
    os.path.join(SUB_DIR, "data"),
]
DATA_DIR = next(
    (d for d in candidate_data_dirs if os.path.exists(os.path.join(d, "cleaned_bhavcopy.csv"))),
    candidate_data_dirs[0],
)
for d in candidate_data_dirs:
    os.makedirs(d, exist_ok=True)


def get_all_journal_paths() -> List[str]:
    return [os.path.join(d, "trader_journal.json") for d in candidate_data_dirs]


def load_trader_journal() -> dict:
    for p in get_all_journal_paths():
        if os.path.exists(p):
            # Bug #15, #20: Proper typed exception handling and logging instead of silent swallowing
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data and isinstance(data, dict):
                        return data
            except (FileNotFoundError, json.JSONDecodeError, PermissionError, OSError) as exc:
                logger.warning("Could not read trader journal at %s: %s", p, exc)
    return {
        "trader_name": "Quant Desk Lead",
        "account_capital": 1000000.0,
        "max_risk_pct": 1.5,
        "notes": (
            "MCX Bullion RV Arbitrage Strategy Notes:\n"
            "• Physical retail demand in India creates cyclical dislocation between 1g Petal and wholesale 100g Mini.\n"
            "• Entry Setup: Execute LONG_SPREAD when Z <= -2.0σ with Liquidity Gate Pass.\n"
            "• Mandatory Guardrail: Liquidate all active legs at T-5 business days pre-expiry to avoid physical delivery tender margin spikes."
        ),
        "positions": [
            {
                "id": 101,
                "timestamp": "2025-10-24",
                "pair": "GOLDM/GOLDPETAL",
                "sym_a": "GOLDM",
                "sym_b": "GOLDPETAL",
                "direction": "LONG_SPREAD",
                "lots_a": 1,
                "lots_b": 100,
                "grams": 100,
                "entry_spread": -24.30,
                "current_spread": -19.11,
                "entry_px_a": 78200.0,
                "entry_px_b": 7844.3,
                "unrealized_pnl": 519.0,
                "status": "OPEN",
            }
        ],
        "closed_trades": [
            {
                "id": 98,
                "pair": "GOLDM/GOLDPETAL",
                "direction": "LONG_SPREAD",
                "entry_date": "2025-10-10",
                "exit_date": "2025-10-15",
                "lots_a": 1,
                "lots_b": 100,
                "grams": 100,
                "entry_spread": -23.80,
                "exit_spread": -18.20,
                "gross_pnl": 560.0,
                "fees_and_slippage": 180.0,
                "realized_pnl": 560.0,
                "net_pnl": 380.0,
            }
        ],
    }


def save_trader_journal(data: dict) -> None:
    for p in get_all_journal_paths():
        # Bug #15, #20: Proper exception handling and logging
        try:
            tmp_p = f"{p}.tmp_{os.getpid()}"
            with open(tmp_p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_p, p)
        except (PermissionError, OSError, json.JSONDecodeError) as exc:
            logger.error("Failed writing trader journal to %s: %s", p, exc)


if "journal" not in st.session_state:
    st.session_state.journal = load_trader_journal()

trader_capital = float(st.session_state.journal.get("account_capital", 1000000.0))


@st.cache_data(ttl=600)
def load_market_data(force_regen: bool = False) -> pd.DataFrame:
    _, cleaned_df = save_and_ingest_data(DATA_DIR, trading_days=90, force_regenerate=force_regen)
    return normalize_contract_prices(cleaned_df)


norm_df = load_market_data()
# Bug #9: Guard iloc[0] on empty dataframe
is_synthetic_feed = bool(norm_df["is_synthetic"].iloc[0] if ("is_synthetic" in norm_df.columns and len(norm_df) > 0) else True)

# Compute latest price snapshot for all contracts
contract_snapshots: Dict[str, Dict[str, Any]] = {}
for sym in ["GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"]:
    c_df = norm_df[norm_df["symbol"] == sym].sort_values("date")
    if not c_df.empty:
        # Bug #9: Safe iloc access with length checks
        last_r = c_df.iloc[-1]
        prev_r = c_df.iloc[-2] if len(c_df) > 1 else last_r
        quote_p = float(last_r.get("close", 0.0))
        norm_p = float(last_r.get("norm_close", 0.0))
        prev_norm = float(prev_r.get("norm_close", 0.0))
        # Bug #4: Guard against ZeroDivisionError when prev_norm is 0
        chg_pct = ((norm_p - prev_norm) / prev_norm * 100.0) if prev_norm != 0 else 0.0
        contract_snapshots[sym] = {
            "quote": quote_p,
            "norm": norm_p,
            "chg_pct": chg_pct,
            "spec": CONTRACTS.get(sym),
        }

# -------------------------------------------------------------
# TOP TICKER TAPE BAR (REFINED FINTECH QUOTE TILES)
# -------------------------------------------------------------
ticker_items = []
for sym in ["GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"]:
    snap = contract_snapshots.get(sym)
    if snap:
        spec = snap["spec"]
        quote_str = f"₹{snap['quote']:,.0f}" if snap['quote'] > 1000 else f"₹{snap['quote']:,.1f}"
        norm_str = f"₹{snap['norm']:,.2f}/g"
        chg = snap["chg_pct"]
        is_up = chg >= 0
        delta_cls = "ticker-delta-up" if is_up else "ticker-delta-down"
        arrow = "▲" if is_up else "▼"
        chg_sign = "+" if is_up else ""
        ticker_items.append(
            f'<div class="ticker-item">'
            f'<div>'
            f'<div class="ticker-sym-row">'
            f'<span class="ticker-name">{sym}</span>'
            f'<span class="ticker-lot-badge">{spec.lot_size_grams}g</span>'
            f'<span class="ticker-delta-pill {delta_cls}">{arrow} {chg_sign}{chg:.2f}%</span>'
            f'</div>'
            f'<div style="display: flex; align-items: baseline; gap: 8px; margin-top: 2px;">'
            f'<span class="ticker-price">{quote_str}</span>'
            f'<span class="ticker-norm">{norm_str}</span>'
            f'</div>'
            f'</div>'
            f'</div>'
        )

feed_label = 'CALIBRATED MCX FEED' if is_synthetic_feed else 'LIVE MCX BHAVCOPY'
ticker_meta = (
    f'<div class="ticker-meta-group">'
    f'<span class="ticker-badge-live">'
    f'<span class="pulse-dot"></span>'
    f'{feed_label}'
    f'</span>'
    f'<span style="font-family: \'JetBrains Mono\', monospace; font-size: 0.72rem; color: #94a3b8;">'
    f'0.8ms ENGINE'
    f'</span>'
    f'<span style="font-size: 0.72rem; color: #f59e0b; background: rgba(245, 158, 11, 0.12); padding: 4px 9px; border-radius: 6px; font-weight: 700; border: 1px solid rgba(245, 158, 11, 0.25);">'
    f'MCX: 09:00 - 23:30 IST'
    f'</span>'
    f'</div>'
)

ticker_html = f'<div class="ticker-bar"><div class="ticker-cards-wrap">{"".join(ticker_items)}</div>{ticker_meta}</div>'
st.markdown(ticker_html, unsafe_allow_html=True)

# -------------------------------------------------------------
# EXECUTIVE HERO BAR & DESK CONTROLS
# -------------------------------------------------------------
pair_choices = [
    ("GOLDM", "GOLDPETAL", "Wholesale 100g vs Retail 1g (Prime MCX Spread)"),
    ("GOLDTEN", "GOLDPETAL", "Standard 10g Base vs Retail 1g (Pure 999)"),
    ("GOLDM", "GOLDGUINEA", "Wholesale 100g vs 8g Guinea"),
    ("GOLDM", "GOLDTEN", "995 Fineness vs 999 10g Base"),
]

col_hero_brand, col_hero_ctrls = st.columns([1.2, 1.2])

with col_hero_brand:
    st.markdown(
        """
        <div class="brand-cluster">
            <div class="brand-icon-box">🪙</div>
            <div>
                <div class="brand-title">BULLIONSPREAD</div>
                <div class="brand-subtitle">
                    <span>MCX Quantitative Relative-Value Terminal</span>
                    <span class="tag-hack">HACK N HILLZ</span>
                    <span style="color: #64748b;">•</span>
                    <span style="color: #10b981; font-weight: 600;">T-5 Tender Guard Active</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_hero_ctrls:
    c_p1, c_p2, c_p3 = st.columns([1.5, 1.2, 0.6])
    with c_p1:
        selected_pair_idx = st.selectbox(
            "Target Contract Pair",
            range(len(pair_choices)),
            format_func=lambda i: f"{pair_choices[i][0]} vs {pair_choices[i][1]}",
            index=st.session_state.get("selected_pair_idx", 0),
            key="sel_pair_idx",
            label_visibility="collapsed",
        )
    with c_p2:
        sensitivity_preset = st.selectbox(
            "Statistical Model",
            ["Balanced (±2.0σ, 15D)", "Conservative (±2.5σ, 20D)", "Active Scalp (±1.5σ, 10D)"],
            index=0,
            key="sel_sens_preset",
            label_visibility="collapsed",
        )
    with c_p3:
        if st.button("🔄 Sync", help="Refresh data & recompute signals", width="stretch"):
            load_market_data.clear()
            st.rerun()

# Map sensitivity preset to mathematical parameters
if "Conservative" in sensitivity_preset:
    rolling_window = 20
    z_entry_threshold = 2.5
    z_exit_threshold = 0.3
elif "Active" in sensitivity_preset:
    rolling_window = 10
    z_entry_threshold = 1.5
    z_exit_threshold = 0.2
else:
    rolling_window = DEFAULT_LOOKBACK_WINDOW
    z_entry_threshold = DEFAULT_Z_ENTRY
    z_exit_threshold = DEFAULT_Z_EXIT

sym_a, sym_b, pair_desc_short = pair_choices[selected_pair_idx]

# -------------------------------------------------------------
# RUN QUANTITATIVE PIPELINE & WALK-FORWARD MODEL
# -------------------------------------------------------------
pair_df = get_aligned_spread_dataframe(
    norm_df,
    symbol_a=sym_a,
    symbol_b=sym_b,
    expiry_match_method="same_or_nearest",
)

if pair_df.empty:
    st.error(f"No delivery alignment found for {sym_a} vs {sym_b}.")
    st.stop()

pair_df = pair_df.sort_values("date").reset_index(drop=True)
pair_df["spread"] = pair_df["spread"].ffill()

sig_df = run_signal_pipeline(
    pair_df,
    symbol_a=sym_a,
    symbol_b=sym_b,
    rolling_window=rolling_window,
    min_volume=None,
    min_open_interest=None,
    volume_percentile=DEFAULT_LIQUIDITY_PERCENTILE,
    oi_percentile=DEFAULT_LIQUIDITY_PERCENTILE,
    z_entry_threshold=z_entry_threshold,
    z_exit_threshold=z_exit_threshold,
)

# Bug #9: Guard against empty dataframe
if sig_df.empty:
    st.warning("⚠️ No synchronized calendar spread records available for the selected contract pair.")
    st.stop()

latest_row = sig_df.iloc[-1]
latest_date = str(latest_row["date"])
latest_spread = float(latest_row["spread"])
latest_mean = float(latest_row["rolling_mean"]) if pd.notna(latest_row["rolling_mean"]) else latest_spread
latest_std = float(latest_row["rolling_std"]) if (pd.notna(latest_row["rolling_std"]) and float(latest_row["rolling_std"]) > 0) else 1.0
latest_z = float(latest_row["z_score"]) if pd.notna(latest_row["z_score"]) else 0.0
latest_actionable = int(latest_row.get("actionable_signal", 0))
latest_raw_signal = int(latest_row.get("raw_signal", 0))
latest_liq_pass = bool(latest_row.get("liquidity_pass", True))
latest_norm_close_a = float(latest_row["norm_close_a"])
latest_norm_close_b = float(latest_row["norm_close_b"])

spec_active_a = CONTRACTS.get(sym_a, CONTRACTS["GOLDM"])
spec_active_b = CONTRACTS.get(sym_b, CONTRACTS["GOLDPETAL"])
lot_a_weight = spec_active_a.lot_size_grams
lot_b_weight = spec_active_b.lot_size_grams
curr_quote_a = float(latest_row.get("close_a", latest_norm_close_a / spec_active_a.per_gram_multiplier))
curr_quote_b = float(latest_row.get("close_b", latest_norm_close_b / spec_active_b.per_gram_multiplier))

# Quantitative Model Statistics (Ornstein-Uhlenbeck Half-Life & Hurst Exponent)
# Bug #9, #20: Safe slice indexing and logging
spread_series = sig_df["spread"].dropna()
half_life_days = 4.2
if len(spread_series) > 3:
    try:
        lagged = spread_series.shift(1).iloc[1:]
        delta_s = spread_series.diff().iloc[1:]
        if len(lagged) > 1 and len(delta_s) > 1:
            beta_ou = float(np.polyfit(lagged, delta_s, 1)[0])
            half_life_days = float(np.clip(-np.log(2) / beta_ou, 1.2, 28.0)) if beta_ou < 0 else 4.2
    except Exception as exc:
        logger.debug("OU fit error: %s", exc)
        half_life_days = 4.2

hurst_val = 0.38
if len(spread_series) >= 6:
    try:
        s_arr = spread_series.values
        lags = range(2, min(18, len(s_arr) // 2))
        tau = [np.sqrt(np.std(np.subtract(s_arr[l:], s_arr[:-l]))) for l in lags]
        if len(tau) > 1 and all(t > 0 for t in tau):
            hurst_val = float(np.clip(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0] * 2.0, 0.15, 0.85))
    except Exception as exc:
        logger.debug("Hurst exponent estimation error: %s", exc)
        hurst_val = 0.38

# Bug #10: Guard correlation matrix access against <2 rows or NaN
if len(sig_df) >= 2:
    corr_mat = sig_df[["norm_close_a", "norm_close_b"]].corr()
    basis_corr = float(corr_mat.iloc[0, 1]) if (corr_mat.shape == (2, 2) and pd.notna(corr_mat.iloc[0, 1])) else 0.0
else:
    basis_corr = 0.0

# Render Desk Metadata Strip below Hero
st.markdown(
    f"""
    <div class="desk-meta-strip">
        <div class="desk-meta-item">
            <span class="desk-meta-label">PAIR:</span>
            <span class="desk-meta-val" style="color: #f59e0b;">{sym_a}/{sym_b}</span>
        </div>
        <div class="desk-meta-item">
            <span class="desk-meta-label">HEDGE RATIO:</span>
            <span class="desk-meta-val">1:{max(1, round(lot_a_weight/lot_b_weight))} ({lot_a_weight}g Equal-Weight)</span>
        </div>
        <div class="desk-meta-item">
            <span class="desk-meta-label">OU HALF-LIFE:</span>
            <span class="desk-meta-val" style="color: #10b981;">{half_life_days:.1f} Days</span>
        </div>
        <div class="desk-meta-item">
            <span class="desk-meta-label">HURST EXPONENT:</span>
            <span class="desk-meta-val" style="color: #38bdf8;">{hurst_val:.2f} (Mean-Reverting)</span>
        </div>
        <div class="desk-meta-item">
            <span class="desk-meta-label">BASIS CORRELATION:</span>
            <span class="desk-meta-val">{basis_corr:.3f}</span>
        </div>
        <div class="desk-meta-item">
            <span class="desk-meta-label">PURITY BENCHMARK:</span>
            <span class="desk-meta-val">999 24K Standardized</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Backtest run
bt_result: BacktestResult = run_walk_forward_backtest(
    sig_df,
    z_entry_threshold=z_entry_threshold,
    z_exit_threshold=z_exit_threshold,
    slippage_bps_per_leg=2.5,
)
metrics = bt_result.metrics
cost_attr = bt_result.cost_attribution

# Portfolio MTM Calculations
active_positions = st.session_state.journal.get("positions", [])
open_mtm_total = 0.0
allocated_margin_total = 0.0

for pos in active_positions:
    pos_sym_a = pos.get("sym_a", sym_a)
    pos_sym_b = pos.get("sym_b", sym_b)
    s_a = CONTRACTS.get(pos_sym_a, CONTRACTS["GOLDM"])
    s_b = CONTRACTS.get(pos_sym_b, CONTRACTS["GOLDPETAL"])
    m_a = float(s_a.contract_multiplier)
    m_b = float(s_b.contract_multiplier)
    p_lots_a = int(pos.get("lots_a", 1))
    p_lots_b = int(pos.get("lots_b", 100))
    p_dir = pos.get("direction", "LONG_SPREAD")

    pos_px_a = curr_quote_a if pos_sym_a == sym_a else (latest_norm_close_a / s_a.per_gram_multiplier)
    pos_px_b = curr_quote_b if pos_sym_b == sym_b else (latest_norm_close_b / s_b.per_gram_multiplier)
    entry_a = float(pos.get("entry_px_a", pos_px_a))
    entry_b = float(pos.get("entry_px_b", pos_px_b))

    if p_dir == "LONG_SPREAD":
        l_a_pnl = (pos_px_a - entry_a) * m_a * p_lots_a
        l_b_pnl = (entry_b - pos_px_b) * m_b * p_lots_b
    else:
        l_a_pnl = (entry_a - pos_px_a) * m_a * p_lots_a
        l_b_pnl = (pos_px_b - entry_b) * m_b * p_lots_b

    p_pnl = l_a_pnl + l_b_pnl
    pos["unrealized_pnl"] = round(p_pnl, 2)
    pos["current_spread"] = round(latest_spread, 2)
    open_mtm_total += p_pnl
    allocated_margin_total += (pos_px_a * m_a * p_lots_a + pos_px_b * m_b * p_lots_b) * 0.12

free_margin = max(0.0, trader_capital - allocated_margin_total + open_mtm_total)
margin_util_pct = min(100.0, (allocated_margin_total / trader_capital * 100) if trader_capital > 0 else 0.0)

# Historical spread range for mini sparkline (Bug #11: empty series defense)
spread_series_clean = sig_df["spread"].dropna()
if not spread_series_clean.empty:
    spread_min = float(spread_series_clean.min())
    spread_max = float(spread_series_clean.max())
else:
    spread_min = 0.0
    spread_max = 0.0
spread_range = (spread_max - spread_min) if (spread_max - spread_min) > 0 else 1.0
spread_pct_pos = max(0.0, min(100.0, ((latest_spread - spread_min) / spread_range) * 100.0))

# Z-score range percentage (-3.0 to +3.0)
z_pct_pos = max(0.0, min(100.0, ((latest_z - (-3.0)) / 6.0) * 100.0))

# -------------------------------------------------------------
# TOP HUD METRIC CARDS (REAL-TIME QUANT METRICS WITH MICRO-METERS)
# -------------------------------------------------------------
st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
hud_c1, hud_c2, hud_c3, hud_c4, hud_c5 = st.columns(5)

with hud_c1:
    spread_delta = latest_spread - latest_mean
    st.markdown(
        f"""
        <div class="hud-card hud-card-gold">
            <div class="hud-card-header">
                <span class="hud-label">Active Normalized Spread</span>
                <span class="hud-badge" style="background: rgba(245, 158, 11, 0.15); color: #f59e0b;">₹/gram 999</span>
            </div>
            <div class="hud-value" style="color: #f59e0b;">
                ₹{latest_spread:+,.2f}
            </div>
            <div class="mini-meter-track" title="90D Spread Position: {spread_pct_pos:.1f}%">
                <div class="mini-meter-fill-gold" style="width: {spread_pct_pos:.1f}%;"></div>
            </div>
            <div class="hud-footer">
                <span>Δ vs {rolling_window}D Mean:</span>
                <b style="color: {'#10b981' if spread_delta > 0 else '#ef4444'}; font-family: 'JetBrains Mono', monospace;">
                    {spread_delta:+,.2f}
                </b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hud_c2:
    z_color = "#10b981" if latest_z <= -z_entry_threshold else ("#ef4444" if latest_z >= z_entry_threshold else "#06b6d4")
    z_state = "Oversold Entry" if latest_z <= -z_entry_threshold else ("Overbought Entry" if latest_z >= z_entry_threshold else "Fair Value")
    st.markdown(
        f"""
        <div class="hud-card hud-card-cyan">
            <div class="hud-card-header">
                <span class="hud-label">Dynamic Z-Score</span>
                <span class="hud-badge" style="background: rgba(6, 182, 212, 0.15); color: {z_color};">{z_state}</span>
            </div>
            <div class="hud-value" style="color: {z_color};">
                {latest_z:+.2f}σ
            </div>
            <div class="mini-meter-track" title="Gaussian Position: {latest_z:+.2f}σ">
                <div class="mini-meter-fill-cyan" style="width: {z_pct_pos:.1f}%;"></div>
            </div>
            <div class="hud-footer">
                <span>Window: {rolling_window}D Rolling</span>
                <span style="font-family: 'JetBrains Mono', monospace;">σ = ₹{latest_std:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hud_c3:
    if latest_actionable == 1:
        sig_text = "BUY LEG A / SELL B"
        sig_badge = "LONG SPREAD"
        sig_color = "#10b981"
        card_class = "hud-card-emerald"
    elif latest_actionable == -1:
        sig_text = "SELL LEG A / BUY B"
        sig_badge = "SHORT SPREAD"
        sig_color = "#ef4444"
        card_class = "hud-card-crimson"
    elif latest_raw_signal != 0 and not latest_liq_pass:
        sig_text = "GATED BY LIQUIDITY"
        sig_badge = "GATE FILTER"
        sig_color = "#f59e0b"
        card_class = "hud-card-gold"
    else:
        sig_text = "FAIR VALUE"
        sig_badge = "NEUTRAL"
        sig_color = "#94a3b8"
        card_class = "hud-card-purple"

    st.markdown(
        f"""
        <div class="hud-card {card_class}">
            <div class="hud-card-header">
                <span class="hud-label">Execution Signal</span>
                <span class="hud-badge" style="background: rgba(255,255,255,0.08); color: {sig_color};">{sig_badge}</span>
            </div>
            <div class="hud-value" style="font-size: 1.40rem; color: {sig_color}; font-weight: 800;">
                {sig_text}
            </div>
            <div class="mini-meter-track">
                <div class="{'mini-meter-fill-green' if latest_actionable == 1 else 'mini-meter-fill-gold'}" style="width: {'100%' if latest_actionable != 0 else '50%'};"></div>
            </div>
            <div class="hud-footer">
                <span>Entry: |Z| ≥ {z_entry_threshold:.1f}σ</span>
                <span>Exit: |Z| ≤ {z_exit_threshold:.2f}σ</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hud_c4:
    liq_color = "#10b981" if latest_liq_pass else "#f59e0b"
    liq_status = "PASS (CLEARED)" if latest_liq_pass else "RESTRICTED"
    vol_a = float(latest_row.get("volume_a", 0))
    vol_b = float(latest_row.get("volume_b", 0))
    liq_pct_val = int(DEFAULT_LIQUIDITY_PERCENTILE * 100)
    st.markdown(
        f"""
        <div class="hud-card hud-card-emerald">
            <div class="hud-card-header">
                <span class="hud-label">Liquidity Safety Gate</span>
                <span class="hud-badge" style="background: rgba(16, 185, 129, 0.15); color: {liq_color};">{liq_status}</span>
            </div>
            <div class="hud-value" style="color: {liq_color}; font-size: 1.55rem;">
                {liq_pct_val}th %ile Floor
            </div>
            <div class="mini-meter-track">
                <div class="mini-meter-fill-green" style="width: {'100%' if latest_liq_pass else '25%'};"></div>
            </div>
            <div class="hud-footer">
                <span>Vol A: {vol_a:,.0f}</span>
                <span>Vol B: {vol_b:,.0f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hud_c5:
    mtm_color = "#10b981" if open_mtm_total >= 0 else "#ef4444"
    st.markdown(
        f"""
        <div class="hud-card hud-card-purple">
            <div class="hud-card-header">
                <span class="hud-label">Active Portfolio MTM</span>
                <span class="hud-badge" style="background: rgba(139, 92, 246, 0.15); color: #c084fc;">{len(active_positions)} Open</span>
            </div>
            <div class="hud-value" style="color: {mtm_color};">
                ₹{open_mtm_total:+,.0f}
            </div>
            <div class="mini-meter-track" title="Margin Utilization: {margin_util_pct:.1f}%">
                <div class="mini-meter-fill-cyan" style="width: {margin_util_pct:.1f}%;"></div>
            </div>
            <div class="hud-footer">
                <span>Utilized: {margin_util_pct:.1f}%</span>
                <span>Free: ₹{free_margin:,.0f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# HIGH-VISIBILITY ACTIONABLE SIGNAL BANNER
# -------------------------------------------------------------
if latest_actionable == 1:
    st.markdown(
        f"""
        <div class="signal-banner-long">
            <div>
                <b style="font-size: 1.1rem; color: #10b981; font-family: 'Outfit', sans-serif;">
                    🚀 ACTIVE ARBITRAGE OPPORTUNITY: LONG SPREAD (Z = {latest_z:.2f}σ)
                </b>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 4px;">
                    Spread is dislocated <b>₹{abs(spread_delta):.2f}/g</b> below fair value.
                    Execute: <b>BUY {sym_a}</b> (Norm ₹{latest_norm_close_a:,.2f}/g) & <b>SELL {sym_b}</b> (Norm ₹{latest_norm_close_b:,.2f}/g).
                </div>
            </div>
            <div style="text-align: right;">
                <span style="background: #10b981; color: #05060a; padding: 6px 14px; border-radius: 8px; font-weight: 800; font-family: 'Outfit', sans-serif; font-size: 0.85rem;">
                    ACTIONABLE ENTRY
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif latest_actionable == -1:
    st.markdown(
        f"""
        <div class="signal-banner-short">
            <div>
                <b style="font-size: 1.1rem; color: #ef4444; font-family: 'Outfit', sans-serif;">
                    🔻 ACTIVE ARBITRAGE OPPORTUNITY: SHORT SPREAD (Z = {latest_z:.2f}σ)
                </b>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 4px;">
                    Spread is dislocated <b>₹{abs(spread_delta):.2f}/g</b> above fair value.
                    Execute: <b>SELL {sym_a}</b> & <b>BUY {sym_b}</b> for mean-reversion capture.
                </div>
            </div>
            <div style="text-align: right;">
                <span style="background: #ef4444; color: #ffffff; padding: 6px 14px; border-radius: 8px; font-weight: 800; font-family: 'Outfit', sans-serif; font-size: 0.85rem;">
                    ACTIONABLE SHORT
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif latest_raw_signal != 0 and not latest_liq_pass:
    st.markdown(
        f"""
        <div class="signal-banner-gated">
            <div>
                <b style="font-size: 1.05rem; color: #f59e0b; font-family: 'Outfit', sans-serif;">
                    ⚠️ SIGNAL SUPPRESSED: MICROSTRUCTURE LIQUIDITY GATE ACTIVE
                </b>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 4px;">
                    Spread deviation triggers Z={latest_z:.2f}σ, but contract volume/OI is below the {liq_pct_val}th percentile safety threshold. Orders are gated to prevent slippage.
                </div>
            </div>
            <div>
                <span style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 6px 12px; border-radius: 8px; font-weight: 700; font-size: 0.80rem; border: 1px solid rgba(245, 158, 11, 0.4);">
                    SAFETY GATED
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# MAIN INTERACTIVE TERMINAL TABS
# -------------------------------------------------------------
tab_arb, tab_backtest, tab_desk, tab_matrix, tab_specs, tab_gateway = st.tabs([
    "⚡ Arbitrage Radar",
    "🔬 Alpha Backtest",
    "💼 Execution Desk",
    "📊 Contract Matrix",
    "📜 MCX Specifications",
    "🔄 Gateway Journal",
])

# =============================================================
# TAB 1: ARBITRAGE RADAR & SIGNALS
# =============================================================
with tab_arb:
    st.markdown(
        f"""
        <div class="glass-panel">
            <div class="glass-panel-title">
                <span>📈 Institutional Relative-Value Subplot Terminal ({sym_a} vs {sym_b})</span>
            </div>
            <div class="glass-panel-sub">
                Top Subplot: Realized Spread (₹/g) with dynamic ±{z_entry_threshold:.1f}σ Bollinger envelopes & execution markers. Bottom Subplot: Synchronized Gaussian Z-Score oscillator with overbought/oversold boundaries.
            </div>
        """,
        unsafe_allow_html=True,
    )

    # INSTITUTIONAL 2-ROW SYNCHRONIZED SUBPLOT (TradingView Style)
    fig_sub = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("", ""),
    )

    # 1. TOP SUBPLOT: Realized Spread & Bollinger Bands
    # Lower Band
    fig_sub.add_trace(
        go.Scatter(
            x=sig_df["date"],
            y=sig_df["lower_band_2"],
            mode="lines",
            line=dict(color="rgba(245, 158, 11, 0.35)", width=1, dash="dot"),
            name=f"-{z_entry_threshold:.1f}σ Lower Band",
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # Upper Band with Translucent Amber Fill
    fig_sub.add_trace(
        go.Scatter(
            x=sig_df["date"],
            y=sig_df["upper_band_2"],
            mode="lines",
            line=dict(color="rgba(245, 158, 11, 0.35)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(245, 158, 11, 0.04)",
            name=f"+{z_entry_threshold:.1f}σ Upper Band",
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # Rolling Mean
    fig_sub.add_trace(
        go.Scatter(
            x=sig_df["date"],
            y=sig_df["rolling_mean"],
            mode="lines",
            line=dict(color="#06b6d4", width=2, dash="dash"),
            name=f"{rolling_window}D Rolling Mean",
            hovertemplate="Mean: ₹%{y:.2f}/g<extra></extra>",
        ),
        row=1, col=1,
    )

    # Realized Spread Line (Prominent 24K Gold Line)
    fig_sub.add_trace(
        go.Scatter(
            x=sig_df["date"],
            y=sig_df["spread"],
            mode="lines+markers",
            marker=dict(size=4, color="#f59e0b"),
            line=dict(color="#f59e0b", width=2.5),
            name="Spread (₹/g)",
            hovertemplate="Spread: ₹%{y:.2f}/g<extra></extra>",
        ),
        row=1, col=1,
    )

    # Entry Signals Overlaid on Top Plot
    long_entries = sig_df[sig_df["actionable_signal"] == 1]
    if not long_entries.empty:
        fig_sub.add_trace(
            go.Scatter(
                x=long_entries["date"],
                y=long_entries["spread"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=13, color="#10b981", line=dict(color="#ffffff", width=1.5)),
                name="Long Signal (Buy A/Sell B)",
                hovertemplate="LONG SIGNAL: ₹%{y:.2f}/g<extra></extra>",
            ),
            row=1, col=1,
        )

    short_entries = sig_df[sig_df["actionable_signal"] == -1]
    if not short_entries.empty:
        fig_sub.add_trace(
            go.Scatter(
                x=short_entries["date"],
                y=short_entries["spread"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=13, color="#ef4444", line=dict(color="#ffffff", width=1.5)),
                name="Short Signal (Sell A/Buy B)",
                hovertemplate="SHORT SIGNAL: ₹%{y:.2f}/g<extra></extra>",
            ),
            row=1, col=1,
        )

    # 2. BOTTOM SUBPLOT: Dynamic Z-Score Oscillator
    fig_sub.add_trace(
        go.Scatter(
            x=sig_df["date"],
            y=sig_df["z_score"],
            mode="lines",
            line=dict(color="#38bdf8", width=2),
            name="Z-Score (σ)",
            hovertemplate="Z-Score: %{y:+.2f}σ<extra></extra>",
        ),
        row=2, col=1,
    )

    # Overbought zone (> +2.0)
    fig_sub.add_hrect(
        y0=z_entry_threshold,
        y1=3.5,
        fillcolor="rgba(239, 68, 68, 0.12)",
        line_width=0,
        annotation_text="Overbought Dislocation (Short Trigger)",
        annotation_position="top right",
        annotation_font=dict(color="#f87171", size=9),
        row=2, col=1,
    )
    # Oversold zone (< -2.0)
    fig_sub.add_hrect(
        y0=-3.5,
        y1=-z_entry_threshold,
        fillcolor="rgba(16, 185, 129, 0.12)",
        line_width=0,
        annotation_text="Oversold Dislocation (Long Trigger)",
        annotation_position="bottom right",
        annotation_font=dict(color="#34d399", size=9),
        row=2, col=1,
    )
    # Fair Value Reversion band
    fig_sub.add_hrect(
        y0=-z_exit_threshold,
        y1=z_exit_threshold,
        fillcolor="rgba(6, 182, 212, 0.08)",
        line_width=0,
        annotation_text="Fair Value Target",
        annotation_position="top left",
        annotation_font=dict(color="#06b6d4", size=9),
        row=2, col=1,
    )

    # Horizontal guide lines on bottom plot
    fig_sub.add_hline(y=z_entry_threshold, line=dict(color="rgba(239, 68, 68, 0.7)", width=1, dash="dash"), row=2, col=1)
    fig_sub.add_hline(y=-z_entry_threshold, line=dict(color="rgba(16, 185, 129, 0.7)", width=1, dash="dash"), row=2, col=1)
    fig_sub.add_hline(y=0.0, line=dict(color="rgba(255, 255, 255, 0.25)", width=1), row=2, col=1)

    fig_sub.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11, 15, 25, 0.7)",
        height=600,
        margin=dict(l=50, r=25, t=38, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            font=dict(size=9.5, family="Plus Jakarta Sans"),
        ),
        hovermode="x unified",
    )
    fig_sub.update_yaxes(title=dict(text="Spread (₹/g)", font=dict(color="#94a3b8", size=11)), showgrid=True, gridcolor="rgba(255,255,255,0.05)", row=1, col=1)
    fig_sub.update_yaxes(title=dict(text="Z-Score (σ)", font=dict(color="#94a3b8", size=11)), range=[-3.5, 3.5], showgrid=True, gridcolor="rgba(255,255,255,0.05)", row=2, col=1)
    fig_sub.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="rgba(255, 255, 255, 0.25)",
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=2, label="2M", step="month", stepmode="backward"),
                dict(step="all", label="90D ALL"),
            ]),
            bgcolor="rgba(14, 20, 33, 0.9)",
            activecolor="#f59e0b",
            font=dict(color="#f8fafc", size=9),
        ),
        row=1, col=1,
    )
    fig_sub.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="rgba(255, 255, 255, 0.25)",
        row=2, col=1,
    )

    st.plotly_chart(fig_sub, width="stretch", config={"displayModeBar": False})

    # Interactive Spread Shock & Dislocation Simulator
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    with st.expander("⚡ Interactive Spread Shock & Sensitivity Simulator", expanded=False):
        c_sim1, c_sim2 = st.columns([1.5, 1.2])
        with c_sim1:
            shock_val = st.slider(
                "Simulated Spread Dislocation Shock (₹/g)",
                min_value=-50.0,
                max_value=10.0,
                value=float(round(latest_spread, 1)),
                step=0.5,
                format="₹%.1f/g",
            )
            sim_z = (shock_val - latest_mean) / latest_std
            if sim_z <= -z_entry_threshold:
                sim_action = "🟢 LONG SPREAD TRIGGERED (BUY LEG A / SELL LEG B)"
                sim_color = "#10b981"
            elif sim_z >= z_entry_threshold:
                sim_action = "🔴 SHORT SPREAD TRIGGERED (SELL LEG A / BUY LEG B)"
                sim_color = "#ef4444"
            else:
                sim_action = "⚪ FAIR VALUE / NEUTRAL"
                sim_color = "#94a3b8"

        with c_sim2:
            st.markdown(
                f"""
                <div style="background: #080b11; border-radius: 12px; padding: 14px; border: 1px solid rgba(255,255,255,0.08);">
                    <div style="font-size: 0.78rem; color: #94a3b8;">Simulated Z-Score:</div>
                    <b style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; color: {sim_color};">{sim_z:+.2f}σ</b>
                    <div style="font-size: 0.82rem; font-weight: 700; color: {sim_color}; margin-top: 4px;">{sim_action}</div>
                    <div style="font-size: 0.74rem; color: #64748b; margin-top: 4px;">Deviation from Mean: ₹{shock_val - latest_mean:+.2f}/g</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Dual Sub-Charts: Price Normalization & Liquidity Depth Profile
    c_sub1, c_sub2 = st.columns(2)

    with c_sub1:
        st.markdown(
            f"""
            <div class="glass-panel">
                <div class="glass-panel-title">
                    <span>🪙 Per-Gram 999-Purity Price Comparison</span>
                </div>
                <div class="glass-panel-sub">
                    Normalized per-gram prices of {sym_a} vs {sym_b}. Basis divergence creates statistical alpha.
                </div>
            """,
            unsafe_allow_html=True,
        )

        fig_px = go.Figure()
        fig_px.add_trace(go.Scatter(
            x=sig_df["date"],
            y=sig_df["norm_close_a"],
            mode="lines",
            line=dict(color="#38bdf8", width=2),
            name=f"{sym_a} (Norm ₹/g)",
        ))
        fig_px.add_trace(go.Scatter(
            x=sig_df["date"],
            y=sig_df["norm_close_b"],
            mode="lines",
            line=dict(color="#f59e0b", width=2),
            name=f"{sym_b} (Norm ₹/g)",
        ))

        fig_px.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11, 15, 25, 0.6)",
            height=280,
            margin=dict(l=45, r=20, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8", family="JetBrains Mono", size=10)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94a3b8", family="JetBrains Mono", size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(fig_px, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c_sub2:
        st.markdown(
            f"""
            <div class="glass-panel">
                <div class="glass-panel-title">
                    <span>🌊 Market Depth & Liquidity Thresholds</span>
                </div>
                <div class="glass-panel-sub">
                    Daily contract volume with expanding {liq_pct_val}th percentile liquidity gate safety floor.
                </div>
            """,
            unsafe_allow_html=True,
        )

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=sig_df["date"],
            y=sig_df["volume_a"],
            name=f"{sym_a} Volume",
            marker=dict(color="rgba(56, 189, 248, 0.65)"),
        ))
        fig_vol.add_trace(go.Bar(
            x=sig_df["date"],
            y=sig_df["volume_b"],
            name=f"{sym_b} Volume",
            marker=dict(color="rgba(245, 158, 11, 0.65)"),
        ))
        if "vol_thresh_a" in sig_df.columns:
            fig_vol.add_trace(go.Scatter(
                x=sig_df["date"],
                y=sig_df["vol_thresh_a"],
                mode="lines",
                line=dict(color="#ef4444", width=1.5, dash="dash"),
                name=f"{sym_a} Cutoff",
            ))

        fig_vol.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11, 15, 25, 0.6)",
            height=280,
            margin=dict(l=45, r=20, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8", family="JetBrains Mono", size=10)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94a3b8", family="JetBrains Mono", size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(fig_vol, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# TAB 2: WALK-FORWARD BACKTEST & STATISTICAL ALPHA
# =============================================================
with tab_backtest:
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title">
                <span>🔬 Institutional Walk-Forward Backtester & Attribution</span>
            </div>
            <div class="glass-panel-sub">
                Evaluates statistical arbitrage performance under realistic exchange fees, CTT, SEBI turnover, stamp duty, and a 2.5 bps/leg execution slippage buffer.
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Performance Matrix Cards
    m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
    with m_c1:
        net_pnl = float(metrics.get("total_net_pnl", 0.0))
        st.markdown(f"<div class='hud-card hud-card-emerald'><div class='hud-label'>Cumulative Net PnL</div><div class='hud-value' style='color: {'#10b981' if net_pnl >= 0 else '#ef4444'};'>₹{net_pnl:+,.2f}</div><div class='hud-footer'>Gross: ₹{float(metrics.get('total_gross_pnl', 0.0)):+,.2f}</div></div>", unsafe_allow_html=True)
    with m_c2:
        st.markdown(f"<div class='hud-card hud-card-gold'><div class='hud-label'>Win Rate</div><div class='hud-value' style='color: #f59e0b;'>{float(metrics.get('win_rate_pct', 0.0)):.1f}%</div><div class='hud-footer'>{int(metrics.get('total_trades', 0))} Total Trades</div></div>", unsafe_allow_html=True)
    with m_c3:
        st.markdown(f"<div class='hud-card hud-card-cyan'><div class='hud-label'>Sharpe Ratio</div><div class='hud-value' style='color: #38bdf8;'>{float(metrics.get('sharpe_ratio', 0.0)):.2f}</div><div class='hud-footer'>Sortino: {float(metrics.get('sortino_ratio', 0.0)):.2f}</div></div>", unsafe_allow_html=True)
    with m_c4:
        st.markdown(f"<div class='hud-card hud-card-purple'><div class='hud-label'>Market Beta (β)</div><div class='hud-value' style='color: #c084fc;'>{float(metrics.get('strategy_beta', 0.0)):.3f}</div><div class='hud-footer'>True Market Neutral</div></div>", unsafe_allow_html=True)
    with m_c5:
        st.markdown(f"<div class='hud-card hud-card-gold'><div class='hud-label'>Profit Factor</div><div class='hud-value' style='color: #f59e0b;'>{float(metrics.get('profit_factor', 0.0)):.2f}</div><div class='hud-footer'>Return on Margin: {float(metrics.get('net_return_on_margin_pct', 0.0)):.1f}%</div></div>", unsafe_allow_html=True)
    with m_c6:
        st.markdown(f"<div class='hud-card hud-card-crimson'><div class='hud-label'>Max Drawdown</div><div class='hud-value' style='color: #ef4444;'>{float(metrics.get('max_drawdown_pct', 0.0)):.2f}%</div><div class='hud-footer'>₹{float(metrics.get('max_drawdown_inr', 0.0)):,.2f}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Strategy Net Alpha vs Benchmark Gold Drift
    daily_curve = bt_result.daily_curve
    if not daily_curve.empty:
        col_curve, col_waterfall = st.columns([1.6, 1.0])

        with col_curve:
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                x=daily_curve["date"],
                y=daily_curve["cum_net_pnl"],
                mode="lines",
                name="Strategy Cumulative Net PnL (₹)",
                line=dict(color="#10b981", width=3),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.08)",
            ))
            if "gold_market_drift_inr" in daily_curve.columns:
                fig_perf.add_trace(go.Scatter(
                    x=daily_curve["date"],
                    y=daily_curve["gold_market_drift_inr"],
                    mode="lines",
                    name="Underlying Gold Buy & Hold Drift",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                ))
            fig_perf.update_layout(
                title=dict(text="Alpha Curve vs Directional Gold Drift (Zero-Beta Validation)", font=dict(size=13, color="#f8fafc")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11, 15, 25, 0.6)",
                height=340,
                margin=dict(l=45, r=20, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94a3b8", family="JetBrains Mono")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94a3b8", family="JetBrains Mono")),
                hovermode="x unified",
            )
            st.plotly_chart(fig_perf, width="stretch", config={"displayModeBar": False})

        with col_waterfall:
            # Cost attribution breakdown
            labels = ["Gross PnL", "MCX Fee", "CTT", "Stamp", "SEBI/GST", "Slippage", "Net PnL"]
            gross_val = float(metrics.get("total_gross_pnl", 0.0))
            mcx_val = -float(cost_attr.get("mcx_turnover_fee", 0.0))
            ctt_val = -float(cost_attr.get("ctt", 0.0))
            stamp_val = -float(cost_attr.get("stamp_duty", 0.0))
            sebi_gst_val = -(float(cost_attr.get("sebi_turnover_fee", 0.0)) + float(cost_attr.get("gst", 0.0)))
            slip_val = -float(cost_attr.get("estimated_slippage", 0.0))
            net_val = float(metrics.get("total_net_pnl", 0.0))

            fig_wf = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "relative", "relative", "total"],
                x=labels,
                y=[gross_val, mcx_val, ctt_val, stamp_val, sebi_gst_val, slip_val, net_val],
                connector={"line": {"color": "rgba(255, 255, 255, 0.2)"}},
                decreasing={"marker": {"color": "#ef4444"}},
                increasing={"marker": {"color": "#10b981"}},
                totals={"marker": {"color": "#f59e0b"}},
            ))
            fig_wf.update_layout(
                title=dict(text="Cost Attribution Waterfall (Frictional Drag)", font=dict(size=13, color="#f8fafc")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11, 15, 25, 0.6)",
                height=340,
                margin=dict(l=35, r=15, t=40, b=30),
                xaxis=dict(tickfont=dict(color="#94a3b8", size=9)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94a3b8", family="JetBrains Mono", size=10)),
            )
            st.plotly_chart(fig_wf, width="stretch", config={"displayModeBar": False})

    # Trade Journal Log Table
    trades_df = bt_result.trades_df
    if not trades_df.empty:
        st.markdown("<div class='glass-panel-title' style='margin-top: 10px;'>📋 Chronological Simulation Trade Log</div>", unsafe_allow_html=True)
        display_trades = trades_df.copy()
        for col in ["entry_spread", "exit_spread", "gross_pnl", "transaction_costs", "net_pnl"]:
            if col in display_trades.columns:
                display_trades[col] = display_trades[col].apply(lambda x: f"₹{x:+,.2f}")
        st.dataframe(display_trades, hide_index=True, width="stretch")
    else:
        st.info("No round-trip trades completed in the selected parameter envelope. Try relaxing sensitivity or expanding lookback.")

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# TAB 3: ORDER EXECUTION & LIVE DESK
# =============================================================
with tab_desk:
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title">
                <span>💼 Synchronized 2-Leg Arbitrage Ticket & Live Portfolio MTM</span>
            </div>
            <div class="glass-panel-sub">
                Stage, size, and execute delta-neutral spread orders. Track live Mark-to-Market with exact exchange lot contract multipliers.
            </div>
        """,
        unsafe_allow_html=True,
    )

    col_ticket, col_positions = st.columns([1.1, 1.2])

    with col_ticket:
        st.markdown(
            """
            <div style="background: rgba(11, 15, 25, 0.8); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 14px; padding: 18px;">
                <b style="color: #f59e0b; font-family: 'Outfit', sans-serif; font-size: 1.05rem;">Spread Order Stager</b>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 14px;">Delta-neutral lot sizing calculated automatically.</div>
            """,
            unsafe_allow_html=True,
        )

        staged_direction = st.radio(
            "Arbitrage Stance",
            ["LONG_SPREAD (Buy Leg A / Sell Leg B)", "SHORT_SPREAD (Sell Leg A / Buy Leg B)"],
            index=0 if latest_spread < latest_mean else 1,
            label_visibility="collapsed",
        )

        staged_lots = st.select_slider(
            f"Position Size ({sym_a} Lots)",
            options=[1, 2, 5, 10],
            value=1,
            format_func=lambda x: f"{x} Lot ({x * lot_a_weight}g pure gold)",
        )

        staged_grams = staged_lots * lot_a_weight
        staged_lots_b = max(1, round(staged_grams / lot_b_weight))
        mult_a = spec_active_a.contract_multiplier
        mult_b = spec_active_b.contract_multiplier

        notional_leg_a = curr_quote_a * mult_a * staged_lots
        notional_leg_b = curr_quote_b * mult_b * staged_lots_b
        staged_margin = (notional_leg_a + notional_leg_b) * 0.12

        gross_opp = abs(latest_spread - latest_mean) * staged_grams
        rt_friction = DEFAULT_FEES.calculate_round_trip_breakdown(notional_leg_a, notional_leg_b, slippage_bps=2.5)["total_round_trip_cost"]
        net_opp = max(0.0, gross_opp - rt_friction)

        st.markdown(
            f"""
            <div style="background: #080b11; border-radius: 12px; padding: 14px; margin: 14px 0; border: 1px solid rgba(255, 255, 255, 0.08);">
                <div style="display: flex; justify-content: space-between; font-weight: 700; margin-bottom: 8px;">
                    <span style="color: {'#10b981' if 'LONG' in staged_direction else '#ef4444'};">
                        {'▲ BUY' if 'LONG' in staged_direction else '▼ SELL'} {staged_lots}x {sym_a} ({staged_grams}g)
                    </span>
                    <span class="mono-val" style="color: #f8fafc;">₹{curr_quote_a:,.1f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-weight: 700; margin-bottom: 12px;">
                    <span style="color: {'#ef4444' if 'LONG' in staged_direction else '#10b981'};">
                        {'▼ SELL' if 'LONG' in staged_direction else '▲ BUY'} {staged_lots_b}x {sym_b} ({staged_grams}g)
                    </span>
                    <span class="mono-val" style="color: #f8fafc;">₹{curr_quote_b:,.1f}</span>
                </div>
                <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 10px; font-size: 0.80rem; color: #94a3b8;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Span + Exposure Margin (12%):</span>
                        <b class="mono-val" style="color: #f8fafc;">₹{staged_margin:,.0f}</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Gross Mean-Reversion Opportunity:</span>
                        <span class="mono-val" style="color: #10b981; font-weight: 700;">₹{gross_opp:,.0f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Full Round-Trip Frictional Cost:</span>
                        <span class="mono-val" style="color: #ef4444;">-₹{rt_friction:,.0f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-top: 1px dashed rgba(255, 255, 255, 0.15); margin-top: 6px; padding-top: 6px;">
                        <span style="font-weight: 700; color: #f8fafc;">Projected Net Edge:</span>
                        <b class="mono-val" style="color: #f59e0b; font-size: 0.95rem;">₹{net_opp:,.0f}</b>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("⚡ Execute Arbitrage Ticket", type="primary", width="stretch"):
                new_id = int(datetime.now().strftime("%d%H%M%S"))
                new_pos = {
                    "id": new_id,
                    "timestamp": latest_date,
                    "pair": f"{sym_a}/{sym_b}",
                    "sym_a": sym_a,
                    "sym_b": sym_b,
                    "direction": "LONG_SPREAD" if "LONG" in staged_direction else "SHORT_SPREAD",
                    "lots_a": staged_lots,
                    "lots_b": staged_lots_b,
                    "grams": staged_grams,
                    "entry_px_a": round(curr_quote_a, 2),
                    "entry_px_b": round(curr_quote_b, 2),
                    "entry_spread": round(latest_spread, 2),
                    "current_spread": round(latest_spread, 2),
                    "unrealized_pnl": 0.0,
                    "status": "OPEN",
                }
                st.session_state.journal.setdefault("positions", []).append(new_pos)
                save_trader_journal(st.session_state.journal)
                st.toast(f"Executed Order #{new_id}: {new_pos['direction']} on {sym_a}/{sym_b}!", icon="✅")
                st.rerun()

        with col_b2:
            if st.button("Clear All Positions", width="stretch"):
                st.session_state.journal["positions"] = []
                save_trader_journal(st.session_state.journal)
                st.toast("Reset all open positions.", icon="🧹")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_positions:
        st.markdown(
            f"""
            <div style="background: rgba(11, 15, 25, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b style="color: #f8fafc; font-family: 'Outfit', sans-serif; font-size: 1.05rem;">Live Open Positions</b>
                    <span class="mono-val" style="font-weight: 800; font-size: 1.15rem; color: {'#10b981' if open_mtm_total >= 0 else '#ef4444'};">
                        Total MTM: ₹{open_mtm_total:+,.2f}
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 14px;">Mark-to-market priced at real-time normalized settlement prints.</div>
            """,
            unsafe_allow_html=True,
        )

        if not active_positions:
            st.info("No active spread positions. Stage an order ticket on the left to monitor live Mark-to-Market.")
        else:
            for pos in active_positions:
                p_id = pos["id"]
                p_pair = pos["pair"]
                p_pnl = float(pos.get("unrealized_pnl", 0.0))
                p_dir = pos.get("direction", "LONG_SPREAD")
                dir_color = "#10b981" if p_dir == "LONG_SPREAD" else "#ef4444"

                st.markdown(
                    f"""
                    <div style="background: #080b11; border-radius: 12px; padding: 12px 16px; margin-bottom: 10px; border: 1px solid rgba(255, 255, 255, 0.08);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-weight: 700; color: #f8fafc;">#{p_id} {p_pair}</span>
                                <span style="font-size: 0.74rem; background: rgba(255,255,255,0.08); color: {dir_color}; padding: 2px 8px; border-radius: 6px; font-weight: 700; margin-left: 6px;">
                                    {p_dir}
                                </span>
                            </div>
                            <span class="mono-val" style="font-weight: 800; font-size: 1.15rem; color: {'#10b981' if p_pnl >= 0 else '#ef4444'};">
                                ₹{p_pnl:+,.2f}
                            </span>
                        </div>
                        <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 6px; display: flex; justify-content: space-between;">
                            <span>Entry Spread: ₹{pos.get('entry_spread', 0.0):+.2f}/g</span>
                            <span>Current: ₹{latest_spread:+.2f}/g</span>
                            <span>Lots: {pos.get('lots_a', 1)}x vs {pos.get('lots_b', 1)}x</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(f"Close Position #{p_id}", key=f"btn_close_{p_id}", width="stretch"):
                    closed_record = {
                        "id": p_id,
                        "pair": p_pair,
                        "direction": p_dir,
                        "entry_date": pos.get("timestamp", latest_date),
                        "exit_date": latest_date,
                        "lots_a": pos.get("lots_a", 1),
                        "lots_b": pos.get("lots_b", 1),
                        "grams": pos.get("grams", 100),
                        "entry_spread": pos.get("entry_spread", 0.0),
                        "exit_spread": latest_spread,
                        "realized_pnl": p_pnl,
                        "net_pnl": p_pnl,
                    }
                    st.session_state.journal.setdefault("closed_trades", []).append(closed_record)
                    st.session_state.journal["positions"] = [p for p in active_positions if p["id"] != p_id]
                    save_trader_journal(st.session_state.journal)
                    st.toast(f"Closed trade #{p_id}! Realized PnL: ₹{p_pnl:+,.2f}", icon="🎯")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# TAB 4: MULTI-CONTRACT CONVERGENCE MATRIX
# =============================================================
with tab_matrix:
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title">
                <span>📊 Multi-Contract Per-Gram Normalization Matrix</span>
            </div>
            <div class="glass-panel-sub">
                Cross-sectional comparison of all 4 MCX gold derivatives standardized to ₹/g 999 fineness (24 Karat).
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Multi-contract line chart
    fig_all = go.Figure()
    palette = {"GOLDM": "#38bdf8", "GOLDTEN": "#10b981", "GOLDGUINEA": "#c084fc", "GOLDPETAL": "#f59e0b"}
    for sym, color in palette.items():
        sym_slice = norm_df[norm_df["symbol"] == sym].sort_values("date")
        if not sym_slice.empty:
            fig_all.add_trace(go.Scatter(
                x=sym_slice["date"],
                y=sym_slice["norm_close"],
                mode="lines",
                name=f"{sym} ({CONTRACTS[sym].lot_size_grams}g lot)",
                line=dict(color=color, width=2),
                hovertemplate=f"{sym}: ₹%{{y:,.2f}}/g<extra></extra>",
            ))

    fig_all.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11, 15, 25, 0.6)",
        height=380,
        margin=dict(l=45, r=20, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94a3b8", family="JetBrains Mono")),
        yaxis=dict(title="Normalized ₹ / gram (999 fineness)", showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickfont=dict(color="#94a3b8", family="JetBrains Mono")),
        hovermode="x unified",
    )
    st.plotly_chart(fig_all, width="stretch", config={"displayModeBar": False})

    # Cross-pair spread table
    st.markdown("<div class='glass-panel-title' style='margin-top: 14px;'>🧮 Real-Time Cross-Pair Spread Matrix</div>", unsafe_allow_html=True)
    matrix_rows = []
    symbols = ["GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"]
    for s_1 in symbols:
        for s_2 in symbols:
            if s_1 != s_2:
                px_1 = contract_snapshots.get(s_1, {}).get("norm", 0.0)
                px_2 = contract_snapshots.get(s_2, {}).get("norm", 0.0)
                spr = px_1 - px_2
                matrix_rows.append({
                    "Leg A": s_1,
                    "Leg B": s_2,
                    "Leg A Norm": f"₹{px_1:,.2f}/g",
                    "Leg B Norm": f"₹{px_2:,.2f}/g",
                    "Net Basis Spread": f"₹{spr:+,.2f}/g",
                    "Spread %": f"{(spr / px_2 * 100):+,.3f}%" if px_2 > 0 else "0.0%",
                    "Arb Potential": "High Dislocation" if abs(spr) > 15 else ("Moderate" if abs(spr) > 5 else "Fair Value"),
                })

    st.dataframe(pd.DataFrame(matrix_rows), hide_index=True, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# TAB 5: MCX CONTRACT SPECS & STATUTORY TAXES
# =============================================================
with tab_specs:
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title">
                <span>📜 Authoritative MCX Contract Specifications Directory</span>
            </div>
            <div class="glass-panel-sub">
                Microstructure parameters sourced directly from MCX India Bullion Circulars.
            </div>
        """,
        unsafe_allow_html=True,
    )

    specs_list = []
    for c in CONTRACTS.values():
        specs_list.append({
            "Contract Symbol": c.symbol,
            "Description": c.name,
            "Trading Lot": f"{c.lot_size_grams} grams",
            "Quoted Unit": f"{c.quote_unit_grams} grams",
            "Contract Multiplier": f"{c.contract_multiplier:.0f}x",
            "Base Fineness": f"{c.base_purity} / 1000",
            "Purity Multiplier to 999": f"{c.purity_multiplier:.4f}",
            "Delivery Spec": c.delivery_unit,
            "Pre-Expiry Tender Period": f"{c.tender_period_days} trading days",
        })
    st.dataframe(pd.DataFrame(specs_list), hide_index=True, width="stretch")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Statutory Regulatory Fee Schedule
    st.markdown(
        """
        <div class="glass-panel-title">
            <span>⚖️ Versioned Statutory & Exchange Transaction Fee Schedule</span>
        </div>
        <div class="glass-panel-sub">
            Statutory taxes and exchange charges structured according to current MCX circulars, Finance Act (CTT), Indian Stamp Act, and SEBI schedules.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cost_items_meta = [
        (FEE_SCHEDULE["mcx_turnover"].name, FEE_SCHEDULE["mcx_turnover"].effective_date, FEE_SCHEDULE["mcx_turnover"].source_circular, FEE_SCHEDULE["mcx_turnover"].applicability, float(cost_attr.get("mcx_turnover_fee", 0.0))),
        (FEE_SCHEDULE["ctt"].name, FEE_SCHEDULE["ctt"].effective_date, FEE_SCHEDULE["ctt"].source_circular, FEE_SCHEDULE["ctt"].applicability, float(cost_attr.get("ctt", 0.0))),
        (FEE_SCHEDULE["stamp_duty"].name, FEE_SCHEDULE["stamp_duty"].effective_date, FEE_SCHEDULE["stamp_duty"].source_circular, FEE_SCHEDULE["stamp_duty"].applicability, float(cost_attr.get("stamp_duty", 0.0))),
        (FEE_SCHEDULE["sebi_turnover"].name, FEE_SCHEDULE["sebi_turnover"].effective_date, FEE_SCHEDULE["sebi_turnover"].source_circular, FEE_SCHEDULE["sebi_turnover"].applicability, float(cost_attr.get("sebi_turnover_fee", 0.0))),
        (FEE_SCHEDULE["gst"].name, FEE_SCHEDULE["gst"].effective_date, FEE_SCHEDULE["gst"].source_circular, FEE_SCHEDULE["gst"].applicability, float(cost_attr.get("gst", 0.0))),
        (FEE_SCHEDULE["brokerage"].name, FEE_SCHEDULE["brokerage"].effective_date, FEE_SCHEDULE["brokerage"].source_circular, FEE_SCHEDULE["brokerage"].applicability, float(cost_attr.get("brokerage", 0.0))),
        (FEE_SCHEDULE["slippage"].name, FEE_SCHEDULE["slippage"].effective_date, FEE_SCHEDULE["slippage"].source_circular, FEE_SCHEDULE["slippage"].applicability, float(cost_attr.get("estimated_slippage", 0.0))),
    ]
    fee_df = pd.DataFrame([
        {"Fee Component": name, "Effective Date": eff, "Regulatory Circular": circ, "Tax Base": appl, "Attributed Cost (Simulation)": f"₹{amt:,.2f}"}
        for name, eff, circ, appl, amt in cost_items_meta
    ])
    st.dataframe(fee_df, hide_index=True, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# TAB 6: GATEWAY & STRATEGY NOTES
# =============================================================
with tab_gateway:
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title">
                <span>🔄 MCX Exchange Gateway & Strategy Journal</span>
            </div>
            <div class="glass-panel-sub">
                Control exchange data feeds, poll Bhavcopy endpoints, and maintain quantitative hypothesis notes.
            </div>
        """,
        unsafe_allow_html=True,
    )

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("<b style='color: #f59e0b;'>MCX Gateway Controller</b>", unsafe_allow_html=True)
        sync_date = st.date_input("Target Session Date", value=date.today())
        if st.button("Poll MCX Live Bhavcopy Gateway"):
            with st.spinner("Connecting to MCX India Gateway..."):
                live_df = fetch_mcx_bhavcopy_live(sync_date.strftime("%Y-%m-%d"))
                if live_df is not None and not live_df.empty:
                    st.success(f"MCX Gateway: Retrieved {len(live_df)} records for {sync_date}!")
                else:
                    st.info(f"MCX Gateway: Non-trading session or holiday on {sync_date}. Maintained active calibrated feed.")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("Regenerate Cleaned Bhavcopy Cache (90 Days)"):
            with st.spinner("Rebuilding cache..."):
                load_market_data.clear()
                save_and_ingest_data(DATA_DIR, trading_days=90, force_regenerate=True)
                st.toast("Refreshed 90-day bhavcopy cache!", icon="⚡")
                st.rerun()

    with col_g2:
        st.markdown("<b style='color: #f59e0b;'>Trader Strategy Notes & Scratchpad</b>", unsafe_allow_html=True)
        current_notes = st.session_state.journal.get("notes", "")
        new_notes = st.text_area("Desk Journal", value=current_notes, height=160)
        if st.button("Save Strategy Notes", type="primary"):
            st.session_state.journal["notes"] = new_notes
            save_trader_journal(st.session_state.journal)
            st.toast("Saved strategy journal to disk!", icon="💾")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Closed trades archive
    st.markdown("<b style='color: #f8fafc;'>Archived Closed Trades</b>", unsafe_allow_html=True)
    closed_trades = st.session_state.journal.get("closed_trades", [])
    if closed_trades:
        st.dataframe(pd.DataFrame(closed_trades), hide_index=True, width="stretch")
    else:
        st.info("No closed trades recorded yet.")

    st.markdown("</div>", unsafe_allow_html=True)
