"""Centralized Trading System Configuration & Market Microstructure Constants.

Authoritative contract specifications sourced from MCX India Bullion Product Specifications:
https://www.mcxindia.com/en/products/bullion/gold

Statutory transaction taxes and exchange charges structured according to current MCX circulars,
Finance Act (CTT), Indian Stamp Act, and SEBI regulatory schedules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    name: str
    lot_size_grams: int          # Trading unit lot size in grams
    quote_unit_grams: int        # Quoted unit in grams (e.g. 10g, 8g, 1g)
    base_purity: int             # Fineness (e.g. 995 or 999)
    standard_purity: int = 999   # Standardized baseline fineness
    tick_size: float = 1.0       # Minimum tick size in INR
    delivery_unit: str = ""      # Physical delivery bar/coin specification
    tender_period_days: int = 5  # Pre-expiry delivery tender period in business days
    valid_from_year: int = 2003  # Inception year on MCX
    official_spec_url: str = "https://www.mcxindia.com/en/products/bullion/gold"

    @property
    def purity_multiplier(self) -> float:
        """Purity adjustment coefficient (standard / base)."""
        return self.standard_purity / self.base_purity

    @property
    def per_gram_multiplier(self) -> float:
        """Multiplier to normalize 1 quoted lot price into 1 gram of 999 purity."""
        return (1.0 / self.quote_unit_grams) * self.purity_multiplier

    @property
    def contract_multiplier(self) -> float:
        """Price multiplier per contract lot: lot_size_grams / quote_unit_grams.
        
        Example:
        - GOLDM: 100g lot / 10g quote = 10 (each ₹1 move in quoted price = ₹10/lot)
        - GOLDTEN: 10g lot / 10g quote = 1 (each ₹1 move in quoted price = ₹1/lot)
        - GOLDGUINEA: 8g lot / 8g quote = 1 (each ₹1 move in quoted price = ₹1/lot)
        - GOLDPETAL: 1g lot / 1g quote = 1 (each ₹1 move in quoted price = ₹1/lot)
        """
        return float(self.lot_size_grams / self.quote_unit_grams)


@dataclass(frozen=True)
class FeeScheduleItem:
    code: str
    name: str
    rate: float
    effective_date: str
    source_circular: str
    applicability: str
    notes: str


# Versioned Statutory & Exchange Regulatory Fee Schedule
FEE_SCHEDULE: Dict[str, FeeScheduleItem] = {
    "mcx_turnover": FeeScheduleItem(
        code="MCX_TURNOVER",
        name="MCX Exchange Turnover Fee",
        rate=0.000021,  # 0.0021% = ₹2.10 per lakh
        effective_date="2023-10-01",
        source_circular="MCX/TRD/678/2023 (Revision of Transaction Charges)",
        applicability="Both buy & sell turnover (gross notional)",
        notes="₹2.10 per lakh of turnover on Group A Non-Agri futures",
    ),
    "sebi_turnover": FeeScheduleItem(
        code="SEBI_TURNOVER",
        name="SEBI Regulatory Fee",
        rate=0.000001,  # ₹10 per crore = 0.0001%
        effective_date="2021-06-01",
        source_circular="SEBI/HO/MIRSD/DOP/CIR/P/2021/573",
        applicability="Both buy & sell turnover (gross notional)",
        notes="₹10 per crore on commodity derivative turnover",
    ),
    "ctt": FeeScheduleItem(
        code="CTT",
        name="Commodities Transaction Tax (CTT)",
        rate=0.00010,  # 0.01%
        effective_date="2013-07-01",
        source_circular="Finance Act 2013 / CBDT Notification No. 43/2013",
        applicability="Sell leg turnover only for non-agricultural futures",
        notes="Levied on the seller at 0.01% of contract settlement value",
    ),
    "stamp_duty": FeeScheduleItem(
        code="STAMP_DUTY",
        name="State Stamp Duty (Uniform Pan-India)",
        rate=0.00002,  # 0.002%
        effective_date="2020-07-01",
        source_circular="Indian Stamp Act 1899 (Part I, Chapter II amended 2019)",
        applicability="Buy leg turnover only",
        notes="0.002% (₹200 per crore) levied on commodity buyers",
    ),
    "gst": FeeScheduleItem(
        code="GST",
        name="Goods and Services Tax (GST)",
        rate=0.18,  # 18%
        effective_date="2017-07-01",
        source_circular="Central GST Act 2017 / Financial Services Schedule",
        applicability="Levied on Brokerage + MCX Exchange Turnover Fee",
        notes="18% service tax on financial transaction charges",
    ),
    "brokerage": FeeScheduleItem(
        code="BROKERAGE",
        name="Institutional DMA Brokerage",
        rate=0.00005,  # 0.5 bps
        effective_date="2024-01-01",
        source_circular="Institutional Member Schedule (Negotiated Prime Brokerage)",
        applicability="Both buy & sell turnover (gross notional)",
        notes="0.5 bps negotiated institutional algorithmic execution fee",
    ),
    "slippage": FeeScheduleItem(
        code="SLIPPAGE",
        name="Execution Slippage Buffer",
        rate=0.00025,  # 2.5 bps per leg
        effective_date="2026-01-01",
        source_circular="Desk Risk Policy 4.2 (Bullion Microstructure Buffer)",
        applicability="Both buy & sell turnover (2.5 bps buffer per leg)",
        notes="Conservative crossing buffer for tight order books",
    ),
}


@dataclass(frozen=True)
class StatutoryFees:
    mcx_turnover_rate: float = FEE_SCHEDULE["mcx_turnover"].rate
    sebi_turnover_rate: float = FEE_SCHEDULE["sebi_turnover"].rate
    ctt_sell_rate: float = FEE_SCHEDULE["ctt"].rate
    stamp_duty_buy_rate: float = FEE_SCHEDULE["stamp_duty"].rate
    gst_rate: float = FEE_SCHEDULE["gst"].rate
    brokerage_rate: float = FEE_SCHEDULE["brokerage"].rate
    default_slippage_bps: float = 2.5

    def calculate_round_trip_breakdown(
        self,
        notional_leg_a: float,
        notional_leg_b: float,
        slippage_bps: float = 2.5,
    ) -> Dict[str, float]:
        """Calculates precise itemized friction for a 2-legged round trip spread."""
        gross_turnover = (notional_leg_a + notional_leg_b) * 2.0
        buy_turnover = notional_leg_a + notional_leg_b
        sell_turnover = notional_leg_a + notional_leg_b

        mcx_charge = gross_turnover * self.mcx_turnover_rate
        sebi_charge = gross_turnover * self.sebi_turnover_rate
        ctt = sell_turnover * self.ctt_sell_rate
        stamp_duty = buy_turnover * self.stamp_duty_buy_rate
        brokerage = gross_turnover * self.brokerage_rate
        gst = (mcx_charge + brokerage) * self.gst_rate
        slippage = gross_turnover * (slippage_bps / 10000.0)

        total_costs = mcx_charge + sebi_charge + ctt + stamp_duty + gst + brokerage + slippage
        cost_bps = (total_costs / gross_turnover) * 10000.0 if gross_turnover > 0 else 0.0

        return {
            "mcx_turnover_fee": round(mcx_charge, 4),
            "sebi_turnover_fee": round(sebi_charge, 4),
            "ctt": round(ctt, 4),
            "stamp_duty": round(stamp_duty, 4),
            "brokerage": round(brokerage, 4),
            "gst": round(gst, 4),
            "estimated_slippage": round(slippage, 4),
            "total_round_trip_cost": round(total_costs, 4),
            "total_friction_bps": round(cost_bps, 2),
        }


# Authoritative MCX Bullion Contract Directory
CONTRACTS: Dict[str, ContractSpec] = {
    "GOLDM": ContractSpec(
        symbol="GOLDM",
        name="Gold Mini",
        lot_size_grams=100,
        quote_unit_grams=10,
        base_purity=995,
        delivery_unit="100 grams bar (995 fineness)",
        tender_period_days=5,
        valid_from_year=2003,
    ),
    "GOLDTEN": ContractSpec(
        symbol="GOLDTEN",
        name="Gold 10 Grams",
        lot_size_grams=10,
        quote_unit_grams=10,
        base_purity=999,
        delivery_unit="10 grams coin / bar (999 fineness)",
        tender_period_days=5,
        valid_from_year=2025,
    ),
    "GOLDGUINEA": ContractSpec(
        symbol="GOLDGUINEA",
        name="Gold Guinea",
        lot_size_grams=8,
        quote_unit_grams=8,
        base_purity=999,
        delivery_unit="8 grams coin (999 fineness)",
        tender_period_days=5,
        valid_from_year=2008,
    ),
    "GOLDPETAL": ContractSpec(
        symbol="GOLDPETAL",
        name="Gold Petal",
        lot_size_grams=1,
        quote_unit_grams=1,
        base_purity=999,
        delivery_unit="1 gram coin / petal (999 fineness)",
        tender_period_days=5,
        valid_from_year=2011,
    ),
}

DEFAULT_FEES = StatutoryFees()
CONTRACT_SPECS = CONTRACTS

# Quantitative Strategy Defaults
DEFAULT_LOOKBACK_WINDOW = 15
DEFAULT_Z_ENTRY = 2.0
DEFAULT_Z_EXIT = 0.25
DEFAULT_STOP_LOSS_Z = 3.8
DEFAULT_EXPIRY_GUARD_DAYS = 5
DEFAULT_LIQUIDITY_PERCENTILE = 0.20
DEFAULT_MIN_VOLUME = 200
DEFAULT_MIN_OI = 500
TARGET_SYMBOLS: List[str] = list(CONTRACTS.keys())
