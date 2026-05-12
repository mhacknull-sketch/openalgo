"""
===============================================================================
          OPTIONS MOMENTUM STRATEGY — Multi-Layer Confirmation
                         OpenAlgo Trading Bot
===============================================================================

Buying NSE F&O options (Calls or Puts) only when institutional-grade
confirmation lines up across FIVE independent signal layers — exactly the
same checks used inside the BuyerEdge analysis tool:

  Layer 1 — Technical Trend          (EMA, VWAP, RSI, MACD on spot candles)
  Layer 2 — OI Flow Intelligence     (PCR, Call/Put Flow, Migration)
  Layer 3 — Greeks Engine            (Delta Imbalance, Gamma Regime)
  Layer 4 — Straddle Engine          (IV Regime, Straddle Velocity)
  Layer 5 — Synthetic Futures        (spot-SF co-movement confirmation)

A composite score is computed (range −100 → +100) along with a trap-risk
score (0 → 100).  An order is placed only when:

  • composite score  ≥ MIN_SCORE  (absolute value, bullish or bearish)
  • trap_score       ≤ MAX_TRAP
  • signal is "EXECUTE" (not "WATCH" or "NO_TRADE")

Run standalone:
    export OPENALGO_API_KEY="your-api-key"
    python options_momentum_strategy.py

Run via OpenAlgo's /python strategy runner:
    OPENALGO_API_KEY            : injected per-strategy.
    OPENALGO_STRATEGY_EXCHANGE  : exchange for this strategy (NFO / BFO).
    HOST_SERVER / WEBSOCKET_URL : inherited from OpenAlgo's .env.
    No code changes required.

⚠ RISK WARNING
    Long options buying has asymmetric payoff but unlimited theta decay.
    Always set PREMIUM_STOP_PCT and ensure adequate capital. This script
    is for educational purposes; backtest before live use.

KEY ENVIRONMENT VARIABLES
    LONG_ONLY_MODE=true        — options buyer mode: Buy CE on bullish signals,
                                 Buy PE on bearish signals.  No short-selling of
                                 options in either direction (default: true).
    BROKER_SL_ORDERS=true      — place exchange-level SELL SL-M at the SL price
                                 and SELL LIMIT at the target price immediately
                                 after each BUY fill (default: true).  The trailing
                                 SL engine modifies the broker SL-M trigger as the
                                 trail ratchets upward.  Software WebSocket monitoring
                                 runs in parallel; on a software-initiated exit the
                                 pending broker orders are cancelled first.
===============================================================================
"""

import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import pandas as pd

from openalgo import api, ta

# ===============================================================================
# CONFIGURATION — all tunable via environment variables
# ===============================================================================

API_KEY  = os.getenv("OPENALGO_API_KEY", "openalgo-apikey")
API_HOST = os.getenv("HOST_SERVER",      "http://127.0.0.1:5000")
WS_URL   = os.getenv("WEBSOCKET_URL",   "ws://127.0.0.1:8765")

# Underlyings to scan
UNDERLYINGS_RAW = os.getenv(
    "UNDERLYINGS",
    "NIFTY,BANKNIFTY,FINNIFTY,RELIANCE,HDFCBANK,ICICIBANK,SBIN,INFY,TCS",
)
UNDERLYINGS = list(set(u.strip() for u in UNDERLYINGS_RAW.split(",") if u.strip()))

# Exchange where these underlyings trade
SPOT_EXCHANGE = os.getenv("EXCHANGE", "NSE")
FNO_EXCHANGE  = os.getenv("FNO_EXCHANGE", "NFO")   # change to BFO for BSE F&O

# Index underlyings whose option chain exchange is NSE_INDEX / BSE_INDEX (not NSE/BSE).
# Used by optionchain(), syntheticfuture() and optiongreeks() calls.
_INDEX_UNDERLYINGS_RAW = os.getenv(
    "INDEX_UNDERLYINGS",
    "NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY,SENSEX,BANKEX,NIFTYNXT50",
)
INDEX_UNDERLYINGS = frozenset(u.strip() for u in _INDEX_UNDERLYINGS_RAW.split(",") if u.strip())
INDEX_EXCHANGE    = os.getenv("INDEX_EXCHANGE", "NSE_INDEX")   # NSE_INDEX or BSE_INDEX

# Telegram username for alerts (without @).  Leave empty to disable.
TELEGRAM_USERNAME = os.getenv("TELEGRAM_USERNAME", "")

# Options parameters
DTE_MIN       = int(os.getenv("DTE_MIN",       "7"))    # min days to expiry
DTE_MAX       = int(os.getenv("DTE_MAX",       "30"))   # max days to expiry
OTM_OFFSET    = int(os.getenv("OTM_OFFSET",    "1"))    # strikes OTM from ATM
LOT_MULTIPLIER= int(os.getenv("LOT_MULTIPLIER","1"))    # lots to buy

# Signal thresholds
# ⚠ TESTING VALUES — restore to MIN_SCORE=50, MAX_TRAP=50 before production.
# MIN_SCORE=15  : fires a trade on almost any non-zero score (stress-test execution path)
# MAX_TRAP=80   : trap gate effectively disabled (all trap scores pass through)
MIN_SCORE      = int(os.getenv("MIN_SCORE",    "15"))    # minimum |score| to trade
MAX_TRAP       = int(os.getenv("MAX_TRAP",     "80"))    # maximum trap score to trade

# Risk Management — Premium SL / Target
PREMIUM_STOP_PCT   = float(os.getenv("PREMIUM_STOP_PCT",    "30.0"))   # % loss from entry premium
PREMIUM_TARGET_PCT = float(os.getenv("PREMIUM_TARGET_PCT",  "80.0"))   # % gain from entry premium

# Session-level risk gates (set to 0 to disable each gate)
MAX_TRADES_PER_SESSION    = int(os.getenv("MAX_TRADES_PER_SESSION",   "5"))   # 0 = unlimited
MAX_CONSECUTIVE_LOSSES    = int(os.getenv("MAX_CONSECUTIVE_LOSSES",   "3"))   # 0 = unlimited
ENTRY_COOLDOWN_SECS       = int(os.getenv("ENTRY_COOLDOWN_SECS",      "300")) # seconds between entries; 0 = none
MAX_DAILY_LOSS_PCT        = float(os.getenv("MAX_DAILY_LOSS_PCT",     "0.0")) # % of account capital; 0 = disabled
MAX_DAILY_LOSS_AMOUNT     = float(os.getenv("MAX_DAILY_LOSS_AMOUNT",  "2000.0")) # absolute ₹ amount; 0 = disabled
RISK_PERCENT              = float(os.getenv("RISK_PERCENT",        "1.0"))    # max premium-risk % per trade

# Trailing SL mode: "spot" (spot-point distance), "premium" (option LTP),
# or "both" (run both; first to trigger exits). Percentages apply to the
# reward distance (entry_premium × PREMIUM_TARGET_PCT / 100) in premium mode.
TRAIL_SL_MODE          = os.getenv("TRAIL_SL_MODE",            "premium")   # "spot" | "premium" | "both"
SPOT_REWARD_PCT        = float(os.getenv("SPOT_REWARD_PCT",        "1.0"))  # % spot move = full reward target
TRAIL_ACTIVATE_AT_PCT  = float(os.getenv("TRAIL_ACTIVATE_AT_PCT",  "25.0")) # activate after 25 % of reward
TRAIL_STEP_RR_PCT      = float(os.getenv("TRAIL_STEP_RR_PCT",      "10.0")) # trail width = reward * this/100

# Long-only: BUY CE on bullish signal, BUY PE on bearish signal. No short selling.
LONG_ONLY_MODE   = os.getenv("LONG_ONLY_MODE",    "true").lower() in ("1", "true", "yes")

# Broker-side protection: place SELL SL-M + SELL LIMIT at the broker immediately after
# each BUY fill. Trailing SL engine modifies the broker SL-M as the trail ratchets up.
# On software-initiated exit, pending broker orders are cancelled first.
BROKER_SL_ORDERS = os.getenv("BROKER_SL_ORDERS",  "true").lower() in ("1", "true", "yes")

# Technicals (spot candles)
CANDLE_INTERVAL  = os.getenv("CANDLE_INTERVAL",  "15m")
LOOKBACK_DAYS    = int(os.getenv("LOOKBACK_DAYS", "5"))
FAST_EMA_PERIOD  = int(os.getenv("FAST_EMA_PERIOD", "9"))
SLOW_EMA_PERIOD  = int(os.getenv("SLOW_EMA_PERIOD", "21"))
RSI_PERIOD       = int(os.getenv("RSI_PERIOD", "14"))

# Loop interval
SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", "60"))  # seconds

# L2 OI/Vol/Premium SMA smoothing: 1 = no smoothing (default); N >= 2 = SMA of
# last N chain snapshots per strike (recommended 3–5, larger = more lag).
# ⚠ Not applied to ATM straddle velocity or delta imbalance (ATM strike can shift).
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "3"))

# ── check_all_checkpoints / best-strike selection ────────────────────────────
# Maximum IVR (IV Rank %) allowed for entry — buyer structural edge degrades
# when options are expensive.  Matches check_all_checkpoints checkpoint 1.
IV_RANK_MAX_ENTRY    = float(os.getenv("IV_RANK_MAX_ENTRY",    "40.0"))

# 52-week IV range for IVR = (current_iv - low) / (high - low) * 100.
# Defaults: India VIX low 8.72 / high 28.91 (May 2026 research). Update quarterly.
IV_52W_LOW  = float(os.getenv("IV_52W_LOW",   "8.72"))   # India VIX 52-wk low  (May 2026)
IV_52W_HIGH = float(os.getenv("IV_52W_HIGH",  "28.91"))  # India VIX 52-wk high (May 2026)

# Liquidity gate — strikes below these thresholds are skipped during strike selection.
MIN_OI_FILTER        = float(os.getenv("MIN_OI_FILTER",        "50000"))  # minimum OI per strike
MIN_VOL_FILTER       = float(os.getenv("MIN_VOL_FILTER",       "10000"))  # minimum volume per strike

# Minimum asymmetry score for strike selection — below this, risk/reward is unattractive.
ASYM_SCORE_THRESHOLD = float(os.getenv("ASYM_SCORE_THRESHOLD", "0.55"))

# When false, skip rather than fall back to OTM offset if no checkpoint strike qualifies.
ALLOW_CHECKPOINT_FALLBACK = os.getenv("ALLOW_CHECKPOINT_FALLBACK", "true").lower() in ("1", "true", "yes")

# Delta target range for strike selection.  Slightly OTM options for long buying
# typically carry a delta of 0.25–0.45 (absolute value).
DELTA_TARGET_LOW     = float(os.getenv("DELTA_TARGET_LOW",     "0.25"))
DELTA_TARGET_HIGH    = float(os.getenv("DELTA_TARGET_HIGH",    "0.45"))

# Order-status polling — tunable for broker-specific fill latency.
# Total wait = ORDER_STATUS_MAX_RETRIES × ORDER_STATUS_POLL_INTERVAL seconds.
ORDER_STATUS_MAX_RETRIES   = int(os.getenv("ORDER_STATUS_MAX_RETRIES",   "15"))  # attempts
ORDER_STATUS_POLL_INTERVAL = float(os.getenv("ORDER_STATUS_POLL_INTERVAL", "2.0"))  # seconds


# ===============================================================================
# STARTUP CONFIGURATION VALIDATION
# ===============================================================================

def _validate_config():
    """
    Check all environment-variable-driven config for sane values.
    Raises SystemExit with a clear error list on any invalid value so the
    operator knows exactly what to fix before the bot starts trading.
    """
    errors: list[str] = []

    if not (0 < PREMIUM_STOP_PCT < 100):
        errors.append(f"PREMIUM_STOP_PCT={PREMIUM_STOP_PCT} must be in range (0, 100)")
    if PREMIUM_TARGET_PCT <= 0:
        errors.append(f"PREMIUM_TARGET_PCT={PREMIUM_TARGET_PCT} must be > 0")
    if RISK_PERCENT <= 0:
        errors.append(f"RISK_PERCENT={RISK_PERCENT} must be > 0")
    if TRAIL_SL_MODE not in ("spot", "premium", "both"):
        errors.append(
            f"TRAIL_SL_MODE={TRAIL_SL_MODE!r} must be one of 'spot', 'premium', 'both'"
        )
    if LOT_MULTIPLIER < 1:
        errors.append(f"LOT_MULTIPLIER={LOT_MULTIPLIER} must be >= 1")
    if not (1 <= MIN_SCORE <= 100):
        errors.append(f"MIN_SCORE={MIN_SCORE} must be in range [1, 100]")
    if not (0 <= MAX_TRAP <= 100):
        errors.append(f"MAX_TRAP={MAX_TRAP} must be in range [0, 100]")
    if DTE_MIN < 0 or DTE_MAX < DTE_MIN:
        errors.append(f"DTE_MIN={DTE_MIN} / DTE_MAX={DTE_MAX}: must satisfy 0 <= DTE_MIN <= DTE_MAX")
    if not (0 < DELTA_TARGET_LOW < DELTA_TARGET_HIGH < 1):
        errors.append(
            f"DELTA_TARGET_LOW={DELTA_TARGET_LOW} / DELTA_TARGET_HIGH={DELTA_TARGET_HIGH}: "
            "must satisfy 0 < low < high < 1"
        )
    if IV_RANK_MAX_ENTRY <= 0 or IV_RANK_MAX_ENTRY > 100:
        errors.append(f"IV_RANK_MAX_ENTRY={IV_RANK_MAX_ENTRY} must be in range (0, 100]")
    if ASYM_SCORE_THRESHOLD <= 0 or ASYM_SCORE_THRESHOLD >= 1:
        errors.append(f"ASYM_SCORE_THRESHOLD={ASYM_SCORE_THRESHOLD} must be in range (0, 1)")
    if ORDER_STATUS_MAX_RETRIES < 1:
        errors.append(f"ORDER_STATUS_MAX_RETRIES={ORDER_STATUS_MAX_RETRIES} must be >= 1")
    if ORDER_STATUS_POLL_INTERVAL < 0:
        errors.append(f"ORDER_STATUS_POLL_INTERVAL={ORDER_STATUS_POLL_INTERVAL} must be >= 0")

    if errors:
        print("[CONFIG] Startup validation failed:")
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(
            "Fix the configuration errors above before running. "
            "See env-var comments at the top of the file."
        )
    print("[CONFIG] All configuration values validated OK")


# ===============================================================================
# TECHNICAL HELPERS (pure-Python, no external TA library required)
# ===============================================================================

def calculate_iv_rank(
    current_iv: float | None,
    iv_52w_low: float | None,
    iv_52w_high: float | None,
) -> float | None:
    """
    Compute IV Rank: (current_iv - low) / (high - low) * 100.
    Returns None when any input is missing or high <= low (disables the gate gracefully).
    """
    if current_iv is None or iv_52w_low is None or iv_52w_high is None:
        return None
    if iv_52w_high <= iv_52w_low:
        return None
    return (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100


# ===============================================================================
# OI / FLOW HELPERS — smoothing + 3-factor Price×Volume×OI classification
# ===============================================================================

def _smooth_chain_rows(history: list) -> list[dict]:
    """
    SMA-smooth per-strike OI/Volume/Premium across N snapshots (oldest-first list).
    Appends six direction fields per row (ce/pe_ltp/vol/oi_dir: +1 rising, -1 falling, 0 flat)
    for the 3-factor classifier. Returns single-bar snapshot unchanged (with zero trend flags).
    """
    if not history:
        return []

    # Single-bar: attach zero-trend flags and return unchanged
    if len(history) == 1:
        result = []
        for row in history[0]:
            r = dict(row)
            r["ce_ltp_dir"] = 0; r["ce_vol_dir"] = 0; r["ce_oi_dir"] = 0
            r["pe_ltp_dir"] = 0; r["pe_vol_dir"] = 0; r["pe_oi_dir"] = 0
            result.append(r)
        return result

    # Build {strike: row} lookup for each snapshot
    snaps = []
    for snap in history:
        d = {}
        for row in snap:
            k = row.get("strike")
            if k is not None:
                d[k] = row
        snaps.append(d)

    all_strikes = sorted({k for s in snaps for k in s})

    SMOOTH_FIELDS = [
        "ce_oi", "pe_oi",
        "ce_volume", "pe_volume",
        "ce_ltp", "pe_ltp",
        "ce_oi_chg", "pe_oi_chg",
        "ce_ltp_chg", "pe_ltp_chg",
        "ce_bid", "ce_ask", "pe_bid", "pe_ask",
    ]

    smoothed = []
    for strike in all_strikes:
        rows = [s[strike] for s in snaps if strike in s]
        if not rows:
            continue

        # Start from the newest snapshot's row (preserves non-numeric fields)
        base = None
        for s in reversed(snaps):
            if strike in s:
                base = dict(s[strike])
                break
        row_out = dict(base)

        # SMA for each numeric field
        for field in SMOOTH_FIELDS:
            vals = [float(r.get(field) or 0) for r in rows]
            row_out[field] = sum(vals) / len(vals)

        # Trend: sign of (newest value − oldest value) for key fields
        def _trend(field: str) -> int:
            oldest = next((s[strike] for s in snaps      if strike in s), None)
            newest = next((s[strike] for s in reversed(snaps) if strike in s), None)
            if oldest is None or newest is None:
                return 0
            diff = float(newest.get(field) or 0) - float(oldest.get(field) or 0)
            return 1 if diff > 0 else (-1 if diff < 0 else 0)

        row_out["ce_ltp_dir"] = _trend("ce_ltp")
        row_out["ce_vol_dir"] = _trend("ce_volume")
        row_out["ce_oi_dir"]  = _trend("ce_oi")
        row_out["pe_ltp_dir"] = _trend("pe_ltp")
        row_out["pe_vol_dir"] = _trend("pe_volume")
        row_out["pe_oi_dir"]  = _trend("pe_oi")

        smoothed.append(row_out)

    return smoothed


def _compute_pcr(chain_rows: list[dict]) -> float:
    """Put-Call Ratio by OI (works on raw or smoothed chain_rows)."""
    ce_oi = sum(r.get("ce_oi", 0) or 0 for r in chain_rows)
    pe_oi = sum(r.get("pe_oi", 0) or 0 for r in chain_rows)
    if ce_oi == 0:
        return 1.0
    return pe_oi / ce_oi


def _call_wall(chain_rows: list[dict]) -> float | None:
    if not chain_rows:
        return None
    return max(chain_rows, key=lambda r: r.get("ce_oi", 0))["strike"]


def _put_wall(chain_rows: list[dict]) -> float | None:
    if not chain_rows:
        return None
    return max(chain_rows, key=lambda r: r.get("pe_oi", 0))["strike"]


def _classify_ce_flow(chain_rows: list[dict]) -> tuple[float, str]:
    """
    3-factor CE flow classifier: Price × Volume × OI (8-state matrix).
    Uses direction fields from _smooth_chain_rows when LOOKBACK_BARS >= 2;
    falls back to 2-factor OI-change × LTP-change on single-bar data.

    Matrix (CE — positive scores = bullish for underlying):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  LTP↑  Vol↑  OI↑  → +2   Call Buying — strong bullish conviction      │
    │  LTP↑  Vol↑  OI↓  → +1   CE Short Covering — moderately bullish       │
    │  LTP↑  Vol↓  OI↑  → +0.5 Low-vol CE accumulation — cautiously bullish │
    │  LTP↑  Vol↓  OI↓  → 0    Fading CE interest — weakening (skip)        │
    │  LTP↓  Vol↑  OI↑  → -2   Call Writing — strong bearish signal         │
    │  LTP↓  Vol↑  OI↓  → -1   CE Long Unwinding — moderately bearish       │
    │  LTP↓  Vol↓  OI↑  → -0.5 Low-vol call writing — cautiously bearish    │
    │  LTP↓  Vol↓  OI↓  → 0    Fading CE pressure — weakening (skip)        │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    # Aggregate trend direction flags (set by _smooth_chain_rows; 0 on single bar)
    def _agg_dir(field: str) -> int:
        raw = sum(r.get(field, 0) or 0 for r in chain_rows)
        return 1 if raw > 0 else (-1 if raw < 0 else 0)

    ce_ltp_dir = _agg_dir("ce_ltp_dir")
    ce_vol_dir = _agg_dir("ce_vol_dir")
    ce_oi_dir  = _agg_dir("ce_oi_dir")

    # 3-factor model (LOOKBACK_BARS >= 2 → vol_dir available)
    if ce_vol_dir != 0:
        l, v, o = ce_ltp_dir, ce_vol_dir, ce_oi_dir
        if   l ==  1 and v ==  1 and o ==  1: return  2.0, "Call Buying — strong bullish conviction"
        elif l ==  1 and v ==  1 and o == -1: return  1.0, "CE Short Covering — moderately bullish"
        elif l ==  1 and v == -1 and o ==  1: return  0.5, "CE accumulation low volume — cautiously bullish"
        elif l ==  1 and v == -1 and o == -1: return  0.0, "CE fading interest — weakening"
        elif l == -1 and v ==  1 and o ==  1: return -2.0, "Call Writing — strong bearish signal"
        elif l == -1 and v ==  1 and o == -1: return -1.0, "CE Long Unwinding — moderately bearish"
        elif l == -1 and v == -1 and o ==  1: return -0.5, "Call writing low volume — cautiously bearish"
        elif l == -1 and v == -1 and o == -1: return  0.0, "CE pressure fading — weakening bearish"
        return 0.0, "CE Neutral"

    # 2-factor fallback (single bar, no volume trend)
    ce_oi_chg  = sum(r.get("ce_oi_chg", 0) or 0 for r in chain_rows)
    ce_ltp_chg = sum(r.get("ce_ltp_chg", 0) or 0 for r in chain_rows)
    if ce_oi_chg > 0 and ce_ltp_chg > 0.5:  return  2, "Call Buying"
    if ce_oi_chg < 0 and ce_ltp_chg > 0.5:  return  1, "CE Short Covering"
    if ce_oi_chg > 0 and ce_ltp_chg < -0.5: return -2, "Call Writing"
    if ce_oi_chg < 0 and ce_ltp_chg < -0.5: return -1, "CE Long Unwinding"
    return 0, "CE Neutral"


def _classify_pe_flow(chain_rows: list[dict]) -> tuple[float, str]:
    """
    3-factor PE flow classifier: Price × Volume × OI (8-state matrix, scores inverted vs CE).
    PE LTP↑ = bearish for underlying; PE LTP↓ = bullish. Falls back to 2-factor on single bar.

    Matrix (PE — positive scores = bullish for underlying):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PE_LTP↑  Vol↑  OI↑  → -2   Put Buying — strong bearish for underlying│
    │  PE_LTP↑  Vol↑  OI↓  → -1   PE Short Covering — moderately bearish    │
    │  PE_LTP↑  Vol↓  OI↑  → -0.5 Low-vol put accumulation — cautiously bear│
    │  PE_LTP↑  Vol↓  OI↓  → 0    Fading put demand — weakening bearish     │
    │  PE_LTP↓  Vol↑  OI↑  → +2   Put Writing — strong bullish for underlying│
    │  PE_LTP↓  Vol↑  OI↓  → +1   PE Long Unwinding — moderately bullish    │
    │  PE_LTP↓  Vol↓  OI↑  → +0.5 Low-vol put writing — cautiously bullish  │
    │  PE_LTP↓  Vol↓  OI↓  → 0    Fading put pressure — weakening bullish   │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    def _agg_dir(field: str) -> int:
        raw = sum(r.get(field, 0) or 0 for r in chain_rows)
        return 1 if raw > 0 else (-1 if raw < 0 else 0)

    pe_ltp_dir = _agg_dir("pe_ltp_dir")
    pe_vol_dir = _agg_dir("pe_vol_dir")
    pe_oi_dir  = _agg_dir("pe_oi_dir")

    # 3-factor model (LOOKBACK_BARS >= 2)
    if pe_vol_dir != 0:
        l, v, o = pe_ltp_dir, pe_vol_dir, pe_oi_dir
        if   l ==  1 and v ==  1 and o ==  1: return -2.0, "Put Buying — strong bearish for underlying"
        elif l ==  1 and v ==  1 and o == -1: return -1.0, "PE Short Covering — moderately bearish"
        elif l ==  1 and v == -1 and o ==  1: return -0.5, "Put accumulation low volume — cautiously bearish"
        elif l ==  1 and v == -1 and o == -1: return  0.0, "PE demand fading — weakening bearish"
        elif l == -1 and v ==  1 and o ==  1: return  2.0, "Put Writing — strong bullish for underlying"
        elif l == -1 and v ==  1 and o == -1: return  1.0, "PE Long Unwinding — moderately bullish"
        elif l == -1 and v == -1 and o ==  1: return  0.5, "Put writing low volume — cautiously bullish"
        elif l == -1 and v == -1 and o == -1: return  0.0, "PE pressure fading — weakening bullish"
        return 0.0, "PE Neutral"

    # 2-factor fallback (single bar)
    pe_oi_chg  = sum(r.get("pe_oi_chg", 0) or 0 for r in chain_rows)
    pe_ltp_chg = sum(r.get("pe_ltp_chg", 0) or 0 for r in chain_rows)
    if pe_oi_chg > 0 and pe_ltp_chg < -0.5: return  2, "Put Writing"
    if pe_oi_chg > 0 and pe_ltp_chg > 0.5:  return -2, "Put Buying"
    if pe_oi_chg < 0 and pe_ltp_chg > 0.5:  return  1, "PE Short Covering"
    if pe_oi_chg < 0 and pe_ltp_chg < -0.5: return -1, "PE Long Unwinding"
    return 0, "PE Neutral"


# ===============================================================================
# SCORING ENGINE  (mirrors BuyerEdge _compute_signal)
# ===============================================================================

def compute_composite_score(
    spot: float,
    df_spot: pd.DataFrame,
    chain_rows: list[dict],
    atm_ce_ltp: float,
    atm_pe_ltp: float,
    iv_rank: float | None,
    straddle_price: float | None,
    prev_straddle_price: float | None,
    sf_ltp: float | None,          # synthetic-future LTP (from syntheticfuture() or futures quote)
    ce_bid: float | None,
    ce_ask: float | None,
    pe_bid: float | None,
    pe_ask: float | None,
    ce_delta: float | None = None,       # actual CE delta from optiongreeks()
    pe_delta: float | None = None,       # actual PE delta from optiongreeks()
    prev_spot: float | None = None,      # previous scan's spot price (for SF co-movement)
    prev_sf_ltp: float | None = None,    # previous scan's synthetic-future price (for co-movement)
) -> dict:
    """
    Compute a composite directional score (−100 → +100) and trap_score (0 → 100).
    Returns dict: score, label, signal, direction, trap_score, trap_reasons, reasons, components.
    """
    components = []
    reasons    = []

    def _dir(s: float) -> str:
        return "bullish" if s > 0 else "bearish" if s < 0 else "neutral"

    # ── LAYER 1: Technical Trend ─────────────────────────────────────────────
    # L1-a: Fast EMA vs Slow EMA (spot trend)
    s1 = 0
    trend_note = "Insufficient candles"
    if df_spot is not None and len(df_spot) >= SLOW_EMA_PERIOD + 2:
        fast = ta.ema(df_spot["close"], period=FAST_EMA_PERIOD)
        slow = ta.ema(df_spot["close"], period=SLOW_EMA_PERIOD)
        if fast.iloc[-2] > slow.iloc[-2] and fast.iloc[-3] <= slow.iloc[-3]:
            s1 = 1
            trend_note = f"Bullish EMA crossover ({FAST_EMA_PERIOD}/{SLOW_EMA_PERIOD})"
        elif fast.iloc[-2] < slow.iloc[-2] and fast.iloc[-3] >= slow.iloc[-3]:
            s1 = -1
            trend_note = f"Bearish EMA crossover ({FAST_EMA_PERIOD}/{SLOW_EMA_PERIOD})"
        elif fast.iloc[-2] > slow.iloc[-2]:
            s1 = 0.5
            trend_note = "Fast EMA above Slow EMA (bullish)"
        elif fast.iloc[-2] < slow.iloc[-2]:
            s1 = -0.5
            trend_note = "Fast EMA below Slow EMA (bearish)"
    components.append({"label": "EMA Trend", "score": s1, "max": 1, "direction": _dir(s1), "note": trend_note})

    # L1-b: RSI (thresholds 53/47 with 50-line partial scores for intraday sensitivity)
    s2 = 0
    rsi_note = "RSI unavailable"
    if df_spot is not None and len(df_spot) >= RSI_PERIOD + 2:
        rsi = ta.rsi(df_spot["close"], period=RSI_PERIOD)
        rsi_val = rsi.iloc[-2]
        if rsi_val > 53:
            s2 = 1
            rsi_note = f"RSI {rsi_val:.1f} — bullish momentum"
        elif rsi_val > 50:
            s2 = 0.5
            rsi_note = f"RSI {rsi_val:.1f} — mild bullish tilt"
        elif rsi_val < 47:
            s2 = -1
            rsi_note = f"RSI {rsi_val:.1f} — bearish momentum"
        elif rsi_val < 50:
            s2 = -0.5
            rsi_note = f"RSI {rsi_val:.1f} — mild bearish tilt"
        else:
            rsi_note = f"RSI {rsi_val:.1f} — exactly neutral (50)"
    components.append({"label": "RSI Momentum", "score": s2, "max": 1, "direction": _dir(s2), "note": rsi_note})

    # L1-c: MACD Histogram
    s3 = 0
    macd_note = "MACD unavailable"
    if df_spot is not None and len(df_spot) >= 35:
        _, _, hist = ta.macd(df_spot["close"])
        h_now  = hist.iloc[-2]
        h_prev = hist.iloc[-3]
        if h_now > 0 and h_now > h_prev:
            s3 = 1
            macd_note = "MACD Histogram expanding positive"
        elif h_now < 0 and h_now < h_prev:
            s3 = -1
            macd_note = "MACD Histogram expanding negative"
        elif h_now > 0:
            s3 = 0.5
            macd_note = "MACD Histogram positive (contracting)"
        elif h_now < 0:
            s3 = -0.5
            macd_note = "MACD Histogram negative (contracting)"
    components.append({"label": "MACD Histogram", "score": s3, "max": 1, "direction": _dir(s3), "note": macd_note})

    # L1-d: Price vs VWAP
    s4 = 0
    vwap_note = "VWAP unavailable"
    if df_spot is not None and len(df_spot) >= 5 and "volume" in df_spot.columns:
        vwap = ta.vwap(df_spot["high"], df_spot["low"], df_spot["close"], df_spot["volume"])
        vwap_val = vwap.iloc[-2]
        if spot > vwap_val:
            s4 = 1
            vwap_note = f"Spot {spot:.1f} above VWAP {vwap_val:.1f}"
        else:
            s4 = -1
            vwap_note = f"Spot {spot:.1f} below VWAP {vwap_val:.1f}"
    components.append({"label": "Spot vs VWAP", "score": s4, "max": 1, "direction": _dir(s4), "note": vwap_note})

    # ── LAYER 2: OI Flow Intelligence ────────────────────────────────────────
    # L2-a: PCR OI Level — follow OI flow direction (NOT contrarian reversal).
    # PCR < 0.9 = CE dominant (bullish); PCR > 1.3 = PE dominant (bearish).
    pcr = _compute_pcr(chain_rows)
    s5 = 0
    if pcr <= 0.6:    s5 = 1     # CE strongly dominant  → bullish
    elif pcr <= 0.9:  s5 = 0.5   # mild CE dominance     → mildly bullish
    elif pcr <= 1.1:  s5 = 0     # near parity           → neutral
    elif pcr <= 1.3:  s5 = -0.5  # mild PE dominance     → mildly bearish
    else:             s5 = -1    # PE strongly dominant  → bearish
    components.append({"label": "PCR OI Level", "score": s5, "max": 1, "direction": _dir(s5), "note": f"PCR OI {pcr:.2f}"})

    # L2-b: Call Flow
    s6, ce_flow_label = _classify_ce_flow(chain_rows)
    components.append({"label": "Call OI Flow", "score": s6, "max": 2, "direction": _dir(s6), "note": ce_flow_label})

    # L2-c: Put Flow  (4-state model — same as BuyerEdge)
    s7, pe_flow_label = _classify_pe_flow(chain_rows)
    components.append({"label": "Put OI Flow", "score": s7, "max": 2, "direction": _dir(s7), "note": pe_flow_label})

    # L2-d: OI Wall position. Call wall above = resistance (bearish); put wall broken
    # below spot = put writers ITM, selling pressure (also bearish). Between walls:
    # bias by proximity — closer to put wall = mild bullish, closer to call = mild bearish.
    s8 = 0
    cw = _call_wall(chain_rows)
    pw = _put_wall(chain_rows)
    wall_note = "OI walls unavailable"
    if cw and pw and spot:
        if spot < cw and spot > pw:
            # Spot between walls — direction from which wall is closer
            if (cw - spot) > (spot - pw):
                s8 = 0.5
                wall_note = f"Spot between walls, near put support {pw:.0f} (call wall {cw:.0f} far) — mild bullish"
            else:
                s8 = -0.5
                wall_note = f"Spot between walls, near call resistance {cw:.0f} (put wall {pw:.0f} far) — mild bearish"
        elif spot >= cw:
            s8 = -1
            wall_note = f"Spot {spot:.0f} at/above call wall {cw:.0f} — overhead resistance, bearish"
        elif spot <= pw:
            # Put wall BROKEN: spot is below max PE OI strike.
            # Put writers at {pw} are in-the-money — they hedge by selling futures.
            # The 'support' narrative is wrong here; treat as bearish breakdown.
            s8 = -1
            wall_note = f"Spot {spot:.0f} below put wall {pw:.0f} — support broken, put writers hedging (bearish)"
    components.append({"label": "OI Wall Position", "score": s8, "max": 1, "direction": _dir(s8), "note": wall_note})

    # ── LAYER 3: Greeks Engine ───────────────────────────────────────────────
    # L3-a: Delta Imbalance
    #
    # ┌────────────────────────────────────────────────────────────────────────┐
    # │  DELTA THEORY — Options Buyer Perspective (Black-Scholes)             │
    # │                                                                        │
    # │  CE Delta : 0  →  +1   (Deep OTM → Deep ITM)                         │
    # │  PE Delta : -1 →   0   (Deep ITM → Deep OTM)                         │
    # │                                                                        │
    # │  At strike K: CE_delta = Φ(d₁) ≈ +0.50 ATM;  PE_delta = Φ(d₁) − 1  │
    # │  Delta sum: di = CE_delta + PE_delta = 2Φ(d₁) − 1                    │
    # │    di > 0 → CE drifting ITM (bullish); di < 0 → PE drifting ITM      │
    # │    Thresholds: ±0.05 full score, ±0.02 partial (NIFTY weekly, 7DTE)  │
    # └────────────────────────────────────────────────────────────────────────┘
    s9 = 0
    di_note = "Delta imbalance unavailable"
    di = 0.0
    _delta_computed = False  # sentinel — prevents LTP fallback when a method fires

    # Method 1 (primary): ATM CE + PE delta sum via optiongreeks().
    # di = CE_delta + PE_delta; positive = CE ITM (bullish), negative = PE ITM (bearish).
    if ce_delta is not None and pe_delta is not None:
        di = ce_delta + pe_delta
        if di >= 0.05:
            s9 = 1;    di_note = f"ATM Δ sum {di:+.3f} — CE ITM, net bullish  (CE {ce_delta:+.3f} / PE {pe_delta:+.3f})"
        elif di >= 0.02:
            s9 = 0.5;  di_note = f"ATM Δ sum {di:+.3f} — mild CE dominance   (CE {ce_delta:+.3f} / PE {pe_delta:+.3f})"
        elif di <= -0.05:
            s9 = -1;   di_note = f"ATM Δ sum {di:+.3f} — PE ITM, net bearish  (CE {ce_delta:+.3f} / PE {pe_delta:+.3f})"
        elif di <= -0.02:
            s9 = -0.5; di_note = f"ATM Δ sum {di:+.3f} — mild PE dominance   (CE {ce_delta:+.3f} / PE {pe_delta:+.3f})"
        else:
            di_note = f"ATM Δ sum {di:+.3f} — balanced (CE {ce_delta:+.3f} / PE {pe_delta:+.3f})"
        _delta_computed = True

    # Method 2 (rejected): OI-weighted chain delta — measures writer hedging
    # (~70–80% NSE OI = writers), duplicates L2 PCR/OI flow, wrong for Greeks layer.

    # LTP fallback (no Greeks data): (CE − PE) / midpoint → range (−2, +2).
    if not _delta_computed and atm_ce_ltp and atm_pe_ltp and atm_pe_ltp > 0:
        di = (atm_ce_ltp - atm_pe_ltp) / ((atm_ce_ltp + atm_pe_ltp) / 2)
        if di >= 0.10:
            s9 = 1;    di_note = f"LTP proxy Δ {di:+.3f} — CE premium heavy (bullish)"
        elif di >= 0.05:
            s9 = 0.5;  di_note = f"LTP proxy Δ {di:+.3f} — mild CE premium"
        elif di <= -0.10:
            s9 = -1;   di_note = f"LTP proxy Δ {di:+.3f} — PE premium heavy (bearish)"
        elif di <= -0.05:
            s9 = -0.5; di_note = f"LTP proxy Δ {di:+.3f} — mild PE premium"
        else:
            di_note = f"LTP proxy Δ {di:+.3f} — balanced"

    components.append({"label": "Greeks Bias (Δ)", "score": s9, "max": 1, "direction": _dir(s9), "note": di_note})

    # L3-b: Gamma Regime — regime context (same as BuyerEdge component 11)
    # No directional vote; used as confidence multiplier later.
    s10 = 0   # always 0 — non-directional
    gamma_note = "Gamma flip unavailable (no GEX data)"
    _is_short_gamma = None   # unknown without GEX
    components.append({"label": "Gamma Regime", "score": s10, "max": 2, "direction": "neutral", "note": gamma_note})

    # ── LAYER 4: Straddle & IV ───────────────────────────────────────────────
    # L4-a: IV Regime (IVR) — <20% full buyer edge, <40% mild, >60% full seller edge.
    s11 = 0
    iv_note = "IVR unavailable"
    if iv_rank is not None:
        if iv_rank < 20:
            s11 = 1
            iv_note = f"IVR {iv_rank:.1f}% — structurally cheap, full buyer edge"
        elif iv_rank < 40:
            s11 = 0.5
            iv_note = f"IVR {iv_rank:.1f}% — moderate, mild buyer edge"
        elif iv_rank > 60:
            s11 = -1
            iv_note = f"IVR {iv_rank:.1f}% — structurally expensive, buyer disadvantage"
        elif iv_rank > 50:
            s11 = -0.5
            iv_note = f"IVR {iv_rank:.1f}% — elevated, mild seller edge"
        else:
            iv_note = f"IVR {iv_rank:.1f}% — neutral zone (40–50%)"
    components.append({"label": "IV Regime (IVR)", "score": s11, "max": 1, "direction": _dir(s11), "note": iv_note})

    # L4-b: Straddle Velocity — ±1.5% full / ±0.5% partial (1-min calibrated).
    s12 = 0
    straddle_note = "Straddle velocity unavailable"
    straddle_vel  = "Flat"
    if straddle_price and prev_straddle_price and prev_straddle_price > 0:
        chg_pct = (straddle_price - prev_straddle_price) / prev_straddle_price * 100
        if chg_pct >= 1.5:
            s12 = 2
            straddle_vel  = "Expanding"
            straddle_note = f"Straddle expanding {chg_pct:+.1f}% — real directional move, buyer edge"
        elif chg_pct >= 0.5:
            s12 = 1
            straddle_vel  = "Mild Expansion"
            straddle_note = f"Straddle mild expansion {chg_pct:+.1f}% — modest premium growth"
        elif chg_pct <= -1.5:
            s12 = -2
            straddle_vel  = "Contracting"
            straddle_note = f"Straddle contracting {chg_pct:+.1f}% — IV crush, avoid naked buying"
        elif chg_pct <= -0.5:
            s12 = -1
            straddle_vel  = "Mild Contraction"
            straddle_note = f"Straddle mild contraction {chg_pct:+.1f}% — premium fading"
        else:
            straddle_note = f"Straddle flat ({chg_pct:+.1f}%)"
    components.append({"label": "Straddle Velocity", "score": s12, "max": 2, "direction": _dir(s12), "note": straddle_note})

    # ── LAYER 5: Synthetic Futures (spot-SF co-movement) ────────────────────
    # Both spot AND SF must move together between scans to score. Raw basis alone
    # (backwardation/carry) gives no directional vote. Wide spread suppresses signal.
    s13 = 0
    sf_note = "SF data unavailable"
    if sf_ltp and spot:
        spread_pct = None
        if ce_bid and ce_ask and ce_bid > 0:
            spread_pct = (ce_ask - ce_bid) / ((ce_ask + ce_bid) / 2) * 100

        if spread_pct is not None and spread_pct > 1.5:
            # Wide bid-ask on the option — executable cost degrades signal
            s13 = 0
            sf_note = f"Wide option spread {spread_pct:.1f}% — executable cost degrades signal"
        elif prev_spot is not None and prev_sf_ltp is not None:
            # Both must exceed 0.03% of spot to confirm co-movement (1-min calibrated).
            move_threshold = spot * 0.0003   # 0.03% of spot (1-min calibrated)
            spot_move = spot - prev_spot
            sf_move   = sf_ltp - prev_sf_ltp
            if spot_move > move_threshold and sf_move > move_threshold:
                s13 = 1
                sf_note = (
                    f"SF co-movement bullish: spot Δ{spot_move:+.1f}, "
                    f"SF Δ{sf_move:+.1f} — confirming"
                )
            elif spot_move < -move_threshold and sf_move < -move_threshold:
                s13 = -1
                sf_note = (
                    f"SF co-movement bearish: spot Δ{spot_move:+.1f}, "
                    f"SF Δ{sf_move:+.1f} — confirming"
                )
            else:
                basis = sf_ltp - spot
                carry = "normal" if basis >= -(spot * 0.001) else "backwardation"
                sf_note = (
                    f"SF diverging or insufficient move — no directional vote "
                    f"(basis {basis:+.1f}, {carry})"
                )
        else:
            # No previous snapshot available — can only note current basis
            basis = sf_ltp - spot
            carry = "normal" if basis >= -(spot * 0.001) else "backwardation"
            sf_note = f"SF snapshot only (no prior bar): basis {basis:+.1f} ({carry}) — score 0"
    components.append({"label": "Synthetic Futures", "score": s13, "max": 1, "direction": _dir(s13), "note": sf_note})

    # ── Trap Score ───────────────────────────────────────────────────────────
    trap_score   = 0
    trap_reasons = []

    # T1: IV crush risk — straddle contracting
    if straddle_vel == "Contracting":
        trap_score += 25
        trap_reasons.append("Straddle contracting — IV crush trap")

    # T2: Expensive options (IVR > 60%)
    if iv_rank is not None and iv_rank > 60:
        trap_score += 20
        trap_reasons.append(f"High IVR {iv_rank:.1f}% — options structurally overpriced")

    # T3: SF basis divergence (informational context — does not suppress direction)
    if sf_ltp and spot and abs(sf_ltp - spot) > spot * 0.015:
        trap_score += 15
        trap_reasons.append(f"SF basis divergence {abs(sf_ltp-spot)/spot*100:.2f}% — possible mispricing")

    # T4: Wide option spread
    if ce_bid and ce_ask and ce_bid > 0:
        sp = (ce_ask - ce_bid) / ((ce_ask + ce_bid) / 2) * 100
        if sp > 1.5:
            trap_score += 15
            trap_reasons.append(f"Wide CE spread {sp:.1f}% — high slippage cost")
    if pe_bid and pe_ask and pe_bid > 0:
        sp = (pe_ask - pe_bid) / ((pe_ask + pe_bid) / 2) * 100
        if sp > 1.5:
            trap_score += 15
            trap_reasons.append(f"Wide PE spread {sp:.1f}% — high slippage cost")

    # T5: PCR extreme — reversal risk
    if pcr > 2.5:
        trap_score += 10
        trap_reasons.append(f"PCR OI {pcr:.2f} — extreme put skew, reversal risk")
    elif pcr < 0.4:
        trap_score += 10
        trap_reasons.append(f"PCR OI {pcr:.2f} — extreme call skew, reversal risk")

    trap_score = min(100, trap_score)

    # ── Final Score ──────────────────────────────────────────────────────────
    # Max achievable raw score = 13 (Gamma Regime is disabled, max=2 but always 0).
    # Component maxes: 1+1+1+1 + 1+2+2+1 + 1+[0] + 1+2 + 1 = 15 declared,
    # but Gamma's max=2 is dead weight — dividing by 15 permanently suppresses
    # all scores by 13% and is the primary reason scores plateau at 0–+20.
    # Denominator corrected to 13 (sum of actually achievable component maxes).
    # When GEX integration is added, restore to 15 by including Gamma's max=2.
    MAX_RAW_SCORE = 13  # 15 total − 2 (Gamma disabled)
    raw_score  = sum(c["score"] for c in components)
    base_score = (raw_score / MAX_RAW_SCORE) * 100
    # No gamma-flip multiplier (no GEX data available in this standalone script)
    final_score = int(max(-100, min(100, base_score)))

    # Collect reasons from significant components
    for c in components:
        if abs(c["score"]) >= (c["max"] * 0.5):
            reasons.append(c["note"])

    abs_score = abs(final_score)
    if trap_score > 75:
        signal = "NO_TRADE"
        if trap_reasons:
            reasons.insert(0, f"⚠ High trap risk: {trap_reasons[0]}")
    elif abs_score >= MIN_SCORE:
        signal = "EXECUTE" if trap_score <= MAX_TRAP else "WATCH"
    elif abs_score >= 30:
        signal = "WATCH"
    else:
        signal = "NO_TRADE"

    label = "Bullish" if final_score > 15 else "Bearish" if final_score < -15 else "Neutral"
    # CE = bullish, PE = bearish, None = truly neutral (skip rather than default to a side)
    if final_score > 0:
        direction: str | None = "CE"
    elif final_score < 0:
        direction = "PE"
    else:
        direction = None

    return {
        "score":       final_score,
        "label":       label,
        "signal":      signal,
        "direction":   direction,
        "trap_score":  trap_score,
        "trap_reasons":trap_reasons,
        "reasons":     list(set(reasons)),
        "components":  components,
    }


# ===============================================================================
# OPTION SELECTION HELPER
# ===============================================================================

def select_option_strike(
    chain_rows: list[dict],
    spot: float,
    option_type: str,   # "CE" or "PE"
    otm_offset: int,
) -> dict | None:
    """
    Pick a slightly OTM strike that is `otm_offset` strikes away from ATM.
    Returns the chain row dict or None if unavailable.
    """
    strikes = sorted(set(r["strike"] for r in chain_rows if "strike" in r))
    if not strikes:
        return None

    # Find ATM
    atm = min(strikes, key=lambda x: abs(x - spot))
    idx = strikes.index(atm)

    if option_type == "CE":
        target_idx = min(idx + otm_offset, len(strikes) - 1)
    else:  # PE
        target_idx = max(idx - otm_offset, 0)

    target_strike = strikes[target_idx]

    # Return the row for this strike + option_type
    for row in chain_rows:
        if row.get("strike") == target_strike:
            if option_type in (row.get("option_type", ""), ""):
                return row

    # Fallback: return any row with this strike
    for row in chain_rows:
        if row.get("strike") == target_strike:
            return row

    return None


# ===============================================================================
# MAIN STRATEGY BOT
# ===============================================================================

class OptionsMomentumBot:
    """Multi-layer options momentum bot. Each underlying scanned independently."""

    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)

        self.strategy_name = os.getenv("STRATEGY_NAME", "OptionsMomentum")

        # {underlying: position dict — see _place_entry for all keys}
        self.positions: dict[str, dict] = {}
        # option-symbol → LTP (for SL/target monitoring)
        self.ltp_map:   dict[str, float] = {}
        # underlying → spot LTP (for spot-based trailing SL)
        self.spot_ltp_map: dict[str, float] = {}
        self.exit_lock  = threading.Lock()
        self.exit_queue: set[str] = set()
        # Guards all reads/writes to position lifecycle (positions dict mutations,
        # session counters, daily P&L) across WebSocket, strategy, and exit threads.
        self.state_lock = threading.Lock()
        # Accepted orders that have not reached a terminal broker status yet.
        # These prevent duplicate entries/exits while orderstatus catches up.
        self.pending_entries: dict[str, dict] = {}
        self.pending_exits: dict[str, dict] = {}
        self.running    = True
        self.stop_event = threading.Event()

        # ── Per-underlying straddle price cache ───────────────────────────────
        # Stores the straddle price from the most recent scan cycle so that
        # straddle velocity can be computed without a local 1%-proxy hack.
        self._prev_straddle: dict[str, float] = {}

        # Per-underlying scan-state: previous spot/SF for s13 co-movement scoring;
        # chain history for L2 OI/Vol/Premium SMA smoothing (ATM uses raw chain).
        self._prev_spot: dict[str, float] = {}   # underlying → previous spot
        self._prev_sf:   dict[str, float] = {}   # underlying → previous SF price
        self._chain_history: dict[str, deque] = {}

        # ── Session / Daily Risk State ────────────────────────────────────────
        self.session_date              = datetime.now().strftime("%Y-%m-%d")
        self.session_trade_count       = 0      # trades taken today
        self.session_consecutive_losses= 0      # current loss streak
        self.last_entry_time: datetime | None = None
        self.daily_pnl                 = 0.0   # realised P&L today (₹)

        print("[BOT] Options Momentum Strategy started")
        print(f"[BOT] Underlyings: {', '.join(UNDERLYINGS)}")
        print(f"[BOT] DTE range: {DTE_MIN}–{DTE_MAX} days | OTM offset: {OTM_OFFSET}")
        print(f"[BOT] Score threshold: {MIN_SCORE} | Max trap: {MAX_TRAP}")
        print(f"[BOT] Premium SL: {PREMIUM_STOP_PCT}% | Target: {PREMIUM_TARGET_PCT}%")
        print(f"[BOT] Risk gates — max trades/session: {MAX_TRADES_PER_SESSION} | "
              f"max loss streak: {MAX_CONSECUTIVE_LOSSES} | cooldown: {ENTRY_COOLDOWN_SECS}s")
        print(f"[BOT] Daily loss limit — {MAX_DAILY_LOSS_PCT}% of live available cash | ₹{MAX_DAILY_LOSS_AMOUNT:.0f} absolute")
        print(f"[BOT] Per-trade risk cap: {RISK_PERCENT}% of live available cash "
              f"| checkpoint fallback: {'enabled' if ALLOW_CHECKPOINT_FALLBACK else 'disabled'}")
        print(f"[BOT] Trail SL mode: {TRAIL_SL_MODE} | reward {SPOT_REWARD_PCT}% spot | "
              f"activates at {TRAIL_ACTIVATE_AT_PCT}% | step {TRAIL_STEP_RR_PCT}% of reward")
        print(f"[BOT] Long-only mode: {'ENABLED — CE (calls) only' if LONG_ONLY_MODE else 'disabled (CE + PE)'}")
        print(f"[BOT] Broker SL orders: {'ENABLED — SL-M + LIMIT at broker after entry' if BROKER_SL_ORDERS else 'disabled (software-only)'}")
        if TELEGRAM_USERNAME:
            print(f"[BOT] Telegram alerts → @{TELEGRAM_USERNAME}")

        # Warn about any broker positions that were open before startup
        self._check_open_positions_on_startup()

    # ── Daily date-reset ──────────────────────────────────────────────────────

    def _maybe_reset_daily_state(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.session_date:
            print(f"[RISK] New trading day {today} — resetting session state")
            self.session_date               = today
            self.session_trade_count        = 0
            self.session_consecutive_losses = 0
            self.daily_pnl                  = 0.0
            self.last_entry_time            = None
            self._prev_straddle.clear()
            self._prev_spot.clear()
            self._prev_sf.clear()
            self._chain_history.clear()

    # ── Telegram Alerts ───────────────────────────────────────────────────────

    def _send_telegram(self, message: str, priority: int = 5):
        """Send a Telegram alert. Silently skipped if TELEGRAM_USERNAME is unset."""
        if not TELEGRAM_USERNAME:
            return
        try:
            self.client.telegram(
                username=TELEGRAM_USERNAME,
                message=message,
                priority=priority,
            )
        except Exception as exc:
            print(f"[TELEGRAM] Alert failed: {exc}")

    # ── Startup Position Check ────────────────────────────────────────────────

    def _check_open_positions_on_startup(self):
        """Warn about open NRML positions found at startup (e.g. after a crash)."""
        try:
            pb = self.client.positionbook()
            if not pb or pb.get("status") != "success":
                return
            open_positions = [
                p for p in pb.get("data", [])
                if int(p.get("quantity", 0) or 0) != 0
                and p.get("product", "").upper() == "NRML"
            ]
            if open_positions:
                print(
                    f"[WARNING] {len(open_positions)} open NRML position(s) detected at startup. "
                    "These are NOT tracked by this session's risk state — "
                    "close them manually if they belong to this strategy."
                )
                for p in open_positions:
                    print(
                        f"  • {p.get('symbol')} | qty={p.get('quantity')} | "
                        f"avg={p.get('average_price')}"
                    )
                self._send_telegram(
                    f"⚠️ {self.strategy_name}: {len(open_positions)} open NRML position(s) detected "
                    "at startup — manual review required.",
                    priority=8,
                )
        except Exception as exc:
            print(f"[STARTUP] positionbook check error: {exc}")

    # ── Underlying exchange helper ─────────────────────────────────────────────

    def _underlying_exchange(self, symbol: str) -> str:
        """Return NSE_INDEX/BSE_INDEX for index underlyings, else SPOT_EXCHANGE."""
        return INDEX_EXCHANGE if symbol in INDEX_UNDERLYINGS else SPOT_EXCHANGE

    def _available_capital(self) -> float:
        """Return live available cash from funds(), falling back to 0.0."""
        try:
            resp = self.client.funds()
            data = resp.get("data", {}) if isinstance(resp, dict) else {}
            for key in ("availablecash", "available_cash", "cash", "available_margin", "net"):
                value = data.get(key)
                if value is None:
                    continue
                capital = float(value)
                if capital > 0:
                    return capital
            print(f"[FUNDS] available cash not found in funds() response: {resp}")
        except Exception as exc:
            print(f"[FUNDS] funds() fetch error: {exc}")
        return 0.0

    # ── Risk Gate ─────────────────────────────────────────────────────────────

    def _check_risk_gates(self) -> tuple[bool, str]:
        """
        Evaluate all session-level risk guards.
        Returns (allowed, reason).  reason is empty string when allowed.

        Guards implemented (mirrors the PineScript risk-management block):
          1. Max trades per session
          2. Max consecutive losses (loss streak)
          3. Entry cooldown (minimum seconds between entries)
          4. Daily loss limit (percentage of live available cash OR absolute ₹ amount)
        """
        self._maybe_reset_daily_state()

        # 1. Session trade count
        if MAX_TRADES_PER_SESSION > 0 and self.session_trade_count >= MAX_TRADES_PER_SESSION:
            return False, (
                f"Max trades/session reached ({self.session_trade_count}/{MAX_TRADES_PER_SESSION})"
            )

        # 2. Consecutive loss streak
        if MAX_CONSECUTIVE_LOSSES > 0 and self.session_consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return False, (
                f"Loss streak limit reached ({self.session_consecutive_losses} consecutive losses)"
            )

        # 3. Entry cooldown
        if ENTRY_COOLDOWN_SECS > 0 and self.last_entry_time is not None:
            elapsed = (datetime.now() - self.last_entry_time).total_seconds()
            if elapsed < ENTRY_COOLDOWN_SECS:
                remaining = int(ENTRY_COOLDOWN_SECS - elapsed)
                return False, f"Entry cooldown active ({remaining}s remaining)"

        # 4a. Daily loss — percentage of account capital
        if MAX_DAILY_LOSS_PCT > 0:
            capital = self._available_capital()
            max_loss_amt = capital * (MAX_DAILY_LOSS_PCT / 100.0)
            if self.daily_pnl <= -max_loss_amt:
                reason = (
                    f"Daily loss limit hit ({MAX_DAILY_LOSS_PCT}% = "
                    f"₹{max_loss_amt:.0f}) | current P&L ₹{self.daily_pnl:.0f}"
                )
                self._send_telegram(f"🚨 {self.strategy_name}: {reason}", priority=9)
                return False, reason

        # 4b. Daily loss — absolute amount
        if MAX_DAILY_LOSS_AMOUNT > 0 and self.daily_pnl <= -MAX_DAILY_LOSS_AMOUNT:
            reason = (
                f"Daily loss limit hit (₹{MAX_DAILY_LOSS_AMOUNT:.0f}) "
                f"| current P&L ₹{self.daily_pnl:.0f}"
            )
            self._send_telegram(f"🚨 {self.strategy_name}: {reason}", priority=9)
            return False, reason

        return True, ""

    # ── WebSocket ─────────────────────────────────────────────────────────────

    def _on_ws_data(self, data: dict):
        """Unified WebSocket dispatcher for option LTPs and spot LTPs."""
        if data.get("type") != "market_data":
            return
        sym = data.get("symbol", "")
        ltp = float(data.get("data", {}).get("ltp", 0) or 0)
        if not ltp:
            return

        with self.state_lock:
            self.ltp_map[sym] = ltp
            positions_snapshot = list(self.positions.items())

        # Process each position against this tick
        for ul, pos in positions_snapshot:
            # ── A. Option LTP (Premium-based SL / Target / Trail) ──────────
            if pos.get("symbol") == sym:
                with self.exit_lock:
                    if sym in self.exit_queue:
                        continue
                    entry = pos["entry_premium"]
                    sl    = pos["sl"]
                    tgt   = pos["tgt"]
                    reason = None

                    if ltp <= sl:
                        reason = f"STOPLOSS HIT (LTP {ltp:.2f} ≤ SL {sl:.2f})"
                    elif ltp >= tgt:
                        reason = f"TARGET HIT (LTP {ltp:.2f} ≥ TGT {tgt:.2f})"
                    elif TRAIL_SL_MODE in ("premium", "both"):
                        reward      = entry * (PREMIUM_TARGET_PCT / 100.0)
                        activate_at = reward * (TRAIL_ACTIVATE_AT_PCT / 100.0)
                        trail_width = reward * (TRAIL_STEP_RR_PCT    / 100.0)
                        move        = ltp - entry

                        if move >= activate_at:
                            if not pos.get("premium_trail_active"):
                                pos["premium_trail_active"] = True
                                pos["premium_trail_peak"]   = ltp
                                pos["premium_trail_sl"]     = round(ltp - trail_width, 2)
                                print(f"[TRAIL-P] {ul} — premium trail activated | peak={ltp:.2f} | sl={pos['premium_trail_sl']:.2f}")
                                self._modify_broker_sl(ul, pos["premium_trail_sl"])
                            elif ltp > pos["premium_trail_peak"]:
                                pos["premium_trail_peak"] = ltp
                                pos["premium_trail_sl"]   = round(ltp - trail_width, 2)
                                print(f"[TRAIL-P] {ul} — trail raised | peak={ltp:.2f} | sl={pos['premium_trail_sl']:.2f}")
                                self._modify_broker_sl(ul, pos["premium_trail_sl"])

                            if ltp <= pos["premium_trail_sl"]:
                                reason = f"PREMIUM TRAILING SL HIT (LTP {ltp:.2f} ≤ trail_sl {pos['premium_trail_sl']:.2f})"

                    if reason:
                        self.exit_queue.add(sym)
                        print(f"\n[ALERT] {ul} {pos['option_type']}: {reason}")
                        threading.Thread(target=self._place_exit, args=(ul, reason), daemon=True).start()

            # ── B. Spot LTP (Spot-based Trailing SL) ───────────────────────
            elif pos.get("spot_symbol") == sym and TRAIL_SL_MODE in ("spot", "both"):
                opt_sym = pos["symbol"]
                with self.exit_lock:
                    if opt_sym in self.exit_queue:
                        continue
                    
                    direction   = pos["option_type"]
                    spot_entry  = pos["spot_entry"]
                    reward_dist = pos["reward_dist"]
                    activate_at = reward_dist * (TRAIL_ACTIVATE_AT_PCT / 100.0)
                    trail_width = reward_dist * (TRAIL_STEP_RR_PCT / 100.0)
                    spot        = ltp
                    reason      = None

                    if direction == "CE":
                        move = spot - spot_entry
                        if move >= activate_at:
                            if not pos["trail_active"]:
                                pos["trail_active"] = True
                                pos["trail_peak"]   = spot
                                pos["trail_sl_spot"]= spot - trail_width
                                print(f"[TRAIL-S] {ul} CE — activated | peak={spot:.1f} | sl_spot={pos['trail_sl_spot']:.1f}")
                            elif spot > pos["trail_peak"]:
                                pos["trail_peak"]    = spot
                                pos["trail_sl_spot"] = spot - trail_width
                                print(f"[TRAIL-S] {ul} CE — raised | peak={spot:.1f} | sl_spot={pos['trail_sl_spot']:.1f}")
                            
                            if spot <= pos["trail_sl_spot"]:
                                reason = f"SPOT TRAILING SL HIT (spot {spot:.1f} ≤ trail_sl {pos['trail_sl_spot']:.1f})"
                    else: # PE
                        move = spot_entry - spot
                        if move >= activate_at:
                            if not pos["trail_active"]:
                                pos["trail_active"] = True
                                pos["trail_peak"]   = spot
                                pos["trail_sl_spot"] = spot + trail_width
                                print(f"[TRAIL-S] {ul} PE — activated | trough={spot:.1f} | sl_spot={pos['trail_sl_spot']:.1f}")
                            elif spot < pos["trail_peak"]:
                                pos["trail_peak"]    = spot
                                pos["trail_sl_spot"] = spot + trail_width
                                print(f"[TRAIL-S] {ul} PE — lowered | trough={spot:.1f} | sl_spot={pos['trail_sl_spot']:.1f}")

                            if spot >= pos["trail_sl_spot"]:
                                reason = f"SPOT TRAILING SL HIT (spot {spot:.1f} ≥ trail_sl {pos['trail_sl_spot']:.1f})"

                    if reason:
                        self.exit_queue.add(opt_sym)
                        print(f"\n[ALERT] {ul} {direction}: {reason}")
                        threading.Thread(target=self._place_exit, args=(ul, reason), daemon=True).start()

    def _ws_thread(self):
        try:
            print("[WS] Connecting...")
            self.client.connect()
            # Initial subscription will be updated as positions open
            print("[WS] Connected")
            while not self.stop_event.is_set():
                time.sleep(1)
        except Exception as exc:
            print(f"[WS ERROR] {exc}")
        finally:
            try:
                self.client.disconnect()
            except Exception:
                pass

    def _subscribe(self, exchange: str, symbol: str):
        """Subscribe to option LTP feed."""
        try:
            self.client.subscribe_ltp(
                [{"exchange": exchange, "symbol": symbol}],
                on_data_received=self._on_ws_data,
            )
            print(f"[WS] Subscribed option {symbol}")
        except Exception as exc:
            print(f"[WS] Subscribe error: {exc}")

    def _subscribe_spot(self, symbol: str):
        """Subscribe to underlying spot LTP feed.

        Uses the correct exchange for the underlying — NSE_INDEX for index
        underlyings (NIFTY, BANKNIFTY, …) and SPOT_EXCHANGE (NSE/BSE) for
        equity underlyings.  Using SPOT_EXCHANGE for indices was incorrect;
        the WebSocket proxy would silently fail to match ticks and the
        spot-based trailing SL would never fire.
        """
        try:
            self.client.subscribe_ltp(
                [{"exchange": self._underlying_exchange(symbol), "symbol": symbol}],
                on_data_received=self._on_ws_data,
            )
            print(f"[WS] Subscribed spot {symbol}")
        except Exception as exc:
            print(f"[WS] Spot subscribe error: {exc}")

    def _unsubscribe(self, exchange: str, symbol: str):
        try:
            self.client.unsubscribe_ltp([{"exchange": exchange, "symbol": symbol}])
        except Exception:
            pass

    def _unsubscribe_spot(self, symbol: str):
        try:
            self.client.unsubscribe_ltp([{"exchange": self._underlying_exchange(symbol), "symbol": symbol}])
        except Exception:
            pass

    # ── Data Fetchers ────────────────────────────────────────────────────────

    def _fetch_spot_candles(self, symbol: str) -> pd.DataFrame | None:
        try:
            end   = datetime.now()
            start = end - timedelta(days=LOOKBACK_DAYS)
            df = self.client.history(
                symbol=symbol,
                exchange=SPOT_EXCHANGE,
                interval=CANDLE_INTERVAL,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
            if df is None or len(df) < SLOW_EMA_PERIOD + 5:
                return None
            return df
        except Exception as exc:
            print(f"[DATA] Candle fetch error for {symbol}: {exc}")
            return None

    def _fetch_option_chain(self, symbol: str, expiry: str | None = None) -> tuple[list[dict], str | None]:
        """
        Fetch the option chain via client.optionchain() and flatten the nested
        CE/PE structure into plain dicts so the rest of the strategy can access
        fields like ``ce_ltp``, ``pe_oi``, ``ce_symbol``, etc. directly.

        Parameters
        ----------
        symbol : underlying trading symbol (e.g. "NIFTY", "RELIANCE")
        expiry : expiry date string pre-selected by _fetch_target_expiry().
                 When provided it is passed to optionchain() so the API returns
                 only the chain for that expiry.  When None the API returns the
                 nearest expiry and that expiry string is read from the response.
                 Passing expiry here fixes the HIGH audit finding: optionchain()
                 requires expiry_date for equity underlyings.

        Returns ``(flat_rows, expiry_date)`` where *expiry_date* is the expiry
        string actually used, or ``None`` when unavailable.
        """
        try:
            ul_exchange = self._underlying_exchange(symbol)
            kwargs: dict = dict(underlying=symbol, exchange=ul_exchange)
            if expiry:
                kwargs["expiry_date"] = expiry
            kwargs["strike_count"] = 8
            raw = self.client.optionchain(**kwargs)
            if not raw:
                return [], None
            if isinstance(raw, dict):
                expiry_date = raw.get("expiry_date")
                nested = raw.get("chain", raw.get("data", []))
            else:
                nested, expiry_date = raw, None

            if not isinstance(nested, list):
                return [], expiry_date

            flat_rows: list[dict] = []
            for entry in nested:
                strike = entry.get("strike")
                if strike is None:
                    continue
                ce = entry.get("ce") or {}
                pe = entry.get("pe") or {}

                ce_ltp  = float(ce.get("ltp")  or 0) or None
                pe_ltp  = float(pe.get("ltp")  or 0) or None
                ce_prev = float(ce.get("prev_close") or 0) or None
                pe_prev = float(pe.get("prev_close") or 0) or None

                flat_rows.append({
                    "strike":     strike,
                    "ce_symbol":  ce.get("symbol"),
                    "pe_symbol":  pe.get("symbol"),
                    "ce_ltp":     ce_ltp,
                    "pe_ltp":     pe_ltp,
                    "ce_bid":     float(ce.get("bid") or 0) or None,
                    "ce_ask":     float(ce.get("ask") or 0) or None,
                    "pe_bid":     float(pe.get("bid") or 0) or None,
                    "pe_ask":     float(pe.get("ask") or 0) or None,
                    "ce_oi":      float(ce.get("oi")  or 0),
                    "pe_oi":      float(pe.get("oi")  or 0),
                    # Volume (used by liquidity filter in _select_best_strike)
                    "ce_volume":  float(ce.get("volume") or 0),
                    "pe_volume":  float(pe.get("volume") or 0),
                    # OI and LTP changes vs previous close
                    "ce_oi_chg":  float(ce.get("oi_change") or 0),
                    "pe_oi_chg":  float(pe.get("oi_change") or 0),
                    "ce_ltp_chg": (ce_ltp - ce_prev) if (ce_ltp and ce_prev) else 0.0,
                    "pe_ltp_chg": (pe_ltp - pe_prev) if (pe_ltp and pe_prev) else 0.0,
                    "lotsize":    ce.get("lotsize") or pe.get("lotsize") or 1,
                })
            return flat_rows, expiry_date
        except Exception as exc:
            print(f"[DATA] Option chain error for {symbol}: {exc}")
            return [], None

    def _fetch_quote(self, symbol: str, exchange: str) -> dict:
        try:
            response = self.client.quotes(symbol=symbol, exchange=exchange) or {}
            if response.get('status') == 'success':
                return response.get('data', {})
            else:
                print(f"[DEBUG] {symbol}@{exchange}: quotes API error: {response}")
                return {}
        except Exception as e:
            print(f"[DEBUG] {symbol}@{exchange}: quotes API exception: {e}")
            return {}

    def _fetch_synthetic_future(self, symbol: str, expiry: str | None) -> float | None:
        """
        Return the synthetic-future price for *symbol*.

        For index underlyings (NIFTY, BANKNIFTY, …) the OpenAlgo
        ``client.syntheticfuture()`` API is used — it computes the fair SF price
        from the ATM call-put parity so there is no need to separately fetch a
        futures quote.

        For equity underlyings the API is unavailable; the method falls back to
        fetching the near-month continuous futures quote directly.

        Returns ``None`` when neither source yields a valid price.
        """
        if symbol in INDEX_UNDERLYINGS and expiry:
            try:
                resp = self.client.syntheticfuture(
                    underlying=symbol,
                    exchange=self._underlying_exchange(symbol),
                    expiry_date=expiry,
                )
                if resp and resp.get("status") == "success":
                    price = float(resp.get("synthetic_future_price") or 0)
                    return price if price else None
            except Exception as exc:
                print(f"[DATA] syntheticfuture error for {symbol}: {exc}")

        # Fallback: raw near-month futures quote using the expiry already in scope.
        # The canonical NSE/BSE futures symbol is <SYMBOL><DDMMMYY>FUT
        # (e.g., "NIFTY25MAY25FUT").  The `expiry` parameter is already in DDMMMYY format
        if expiry:
            fut_symbol = f"{symbol}{expiry}FUT"
            sf_q = self._fetch_quote(fut_symbol, FNO_EXCHANGE)
            ltp  = float(sf_q.get("ltp", 0) or 0)
            if ltp:
                return ltp
            print(f"[DATA] syntheticfuture fallback: {fut_symbol} returned no LTP")
        return None

    def _fetch_atm_greeks(
        self,
        symbol: str,
        ce_symbol: str | None,
        pe_symbol: str | None,
    ) -> tuple[float | None, float | None]:
        """
        Fetch the actual ATM delta values from the OpenAlgo ``client.optiongreeks()``
        API for both the CE and PE legs.

        Returns ``(ce_delta, pe_delta)`` where *pe_delta* is negative (puts have
        negative delta).  Either value is ``None`` when the API call fails or the
        symbol is unavailable.

        The delta skew (``ce_delta + pe_delta``) is used in ``compute_composite_score``
        as a more accurate delta-imbalance signal than the LTP-ratio proxy.
        """
        ul_exchange = self._underlying_exchange(symbol)
        ce_delta: float | None = None
        pe_delta: float | None = None
        for opt_sym, key in ((ce_symbol, "ce"), (pe_symbol, "pe")):
            if not opt_sym:
                continue
            try:
                resp = self.client.optiongreeks(
                    symbol=opt_sym,
                    exchange=FNO_EXCHANGE,
                    underlying_symbol=symbol,
                    underlying_exchange=ul_exchange,
                )
                if resp and resp.get("status") == "success":
                    delta = resp.get("greeks", {}).get("delta")
                    if delta is not None:
                        if key == "ce":
                            ce_delta = float(delta)
                        else:
                            pe_delta = float(delta)
            except Exception as exc:
                print(f"[DATA] optiongreeks error for {opt_sym}: {exc}")
        return ce_delta, pe_delta

    def _fetch_option_delta(self, symbol: str, option_symbol: str | None) -> float | None:
        """
        Fetch absolute delta for a candidate option strike. Used by the strike
        filter so DELTA_TARGET_LOW/HIGH are real Greeks gates when optiongreeks()
        is available.
        """
        if not option_symbol:
            return None
        try:
            resp = self.client.optiongreeks(
                symbol=option_symbol,
                exchange=FNO_EXCHANGE,
                underlying_symbol=symbol,
                underlying_exchange=self._underlying_exchange(symbol),
            )
            if resp and resp.get("status") == "success":
                delta = resp.get("greeks", {}).get("delta")
                if delta is not None:
                    return abs(float(delta))
        except Exception as exc:
            print(f"[DATA] option delta error for {option_symbol}: {exc}")
        return None

    def _select_best_strike(
        self,
        symbol: str,
        chain_rows: list[dict],
        spot: float,
        direction: str,   # "CE" or "PE"
        iv_rank: float | None,
    ) -> dict | None:
        """
        Select the best entry strike using the check_all_checkpoints criteria:

          1. Liquidity gate — OI > MIN_OI_FILTER and volume > MIN_VOL_FILTER
             (volume gracefully skipped when the chain doesn't carry it).
          2. Strike range — CE: ATM to +5% OTM; PE: -5% OTM to ATM.
             This approximates the DELTA_TARGET_LOW/HIGH window (0.25–0.45)
             without requiring a per-strike optiongreeks() round-trip.
          3. Asymmetry score gate — overall risk/reward composite must exceed
             ASYM_SCORE_THRESHOLD before committing capital.

        The asymmetry score combines:
          • IV-regime edge  (IVR below 40% → options structurally cheap)
          • OI concentration at the strike (larger OI → more institutional interest)
          • Placeholder weights for catalyst + volume spurt (always 1.0 until
            those data sources are wired in)

        Returns the best chain row dict, or None if no qualifying strike found.
        The fallback logic then uses OTM_OFFSET to pick a strike so the caller
        always has a result when check_all_checkpoints criteria cannot be met.

        OpenAlgo option chain docs:
          https://docs.openalgo.in/api-documentation/v1/data-api/option-chain
        """
        if not chain_rows or not spot:
            return None
        if iv_rank is not None and iv_rank >= IV_RANK_MAX_ENTRY:
            print(f"[STRIKE] IVR {iv_rank:.1f}% >= max {IV_RANK_MAX_ENTRY:.1f}% - buyer edge rejected")
            return None

        # ── 1. Strike range proxy for target delta (0.25–0.45 absolute) ──────
        # Slightly OTM CE: between ATM (spot) and 5% above spot
        # Slightly OTM PE: between 5% below spot and ATM (spot)
        if direction == "CE":
            lo, hi = spot, spot * 1.05
        else:  # PE
            lo, hi = spot * 0.95, spot

        # ── 2. Liquidity + range filter ───────────────────────────────────────
        oi_key  = "ce_oi"   if direction == "CE" else "pe_oi"
        vol_key = "ce_volume" if direction == "CE" else "pe_volume"

        candidates: list[dict] = []
        for row in chain_rows:
            strike = row.get("strike", 0)
            if not (lo <= strike <= hi):
                continue
            oi = float(row.get(oi_key, 0) or 0)
            if oi < MIN_OI_FILTER:
                continue
            vol = float(row.get(vol_key, 0) or 0)
            if vol > 0 and vol < MIN_VOL_FILTER:
                # Only gate on volume when the chain actually carries it
                continue
            candidates.append(row)

        if not candidates:
            return None

        opt_key = "ce_symbol" if direction == "CE" else "pe_symbol"
        delta_checked: list[dict] = []
        delta_available = False
        for row in candidates:
            abs_delta = self._fetch_option_delta(symbol, row.get(opt_key))
            if abs_delta is None:
                continue
            delta_available = True
            if DELTA_TARGET_LOW <= abs_delta <= DELTA_TARGET_HIGH:
                row = dict(row)
                row["_abs_delta"] = abs_delta
                delta_checked.append(row)
        if delta_available:
            if not delta_checked:
                print(
                    f"[STRIKE] No candidate delta in target range "
                    f"{DELTA_TARGET_LOW:.2f}-{DELTA_TARGET_HIGH:.2f}"
                )
                return None
            candidates = delta_checked

        # ── 3. Asymmetry score ────────────────────────────────────────────────
        # Weights from check_all_checkpoints (total = 1.0):
        #   35% IV regime (cheap vs expensive)
        #   25% OI flow (already scored in composite — here use OI magnitude)
        #   20% OI concentration at this strike
        #   10% catalyst (placeholder: always 1.0)
        #   10% volume spurt / delivery (placeholder: always 1.0)
        ivr   = iv_rank if iv_rank is not None else 50.0   # assume moderate when unknown
        total_oi = sum(float(r.get(oi_key, 0) or 0) for r in chain_rows) or 1.0

        best_row: dict | None = None
        best_score = -1.0

        for row in candidates:
            strike_oi  = float(row.get(oi_key, 0) or 0)
            oi_conc    = min(strike_oi / total_oi, 1.0)   # 0..1, higher = more concentrated
            asym_score = (
                (1 - ivr / 100) * 0.35 +   # IV regime edge
                oi_conc           * 0.25 +  # OI concentration (proxy for OI flow weight)
                oi_conc           * 0.20 +  # OI concentration at strike
                1.0               * 0.10 +  # catalyst (placeholder)
                1.0               * 0.10    # volume spurt / delivery (placeholder)
            )
            if asym_score > best_score:
                best_score = asym_score
                best_row   = row

        if best_score < ASYM_SCORE_THRESHOLD:
            print(
                f"[STRIKE] Best asymmetry score {best_score:.3f} < threshold "
                f"{ASYM_SCORE_THRESHOLD} — no qualifying strike"
            )
            return None

        return best_row

    def _fetch_iv_rank(self, spot_quote: dict) -> float | None:
        """
        Derive IV Rank from the already-fetched spot quote dict.

        Accepts the ``spot_quote`` dict returned by ``_fetch_quote()``
        (i.e. the inner ``data`` field of the quotes API response) so that
        no additional ``client.quotes()`` round-trip is needed — the caller
        in ``_scan_underlying`` already has this data.

        The 'iv' field in the quote is the ATM implied volatility (annualised %).
        It is normalised against the India VIX 52-week range (IV_52W_LOW /
        IV_52W_HIGH) to produce an IV Rank in [0, 100].

        Returns None when 'iv' is absent, zero, or not finite.
        """
        try:
            atm_iv = spot_quote.get("iv")
            if atm_iv is None:
                return None
            current_iv = float(atm_iv)
            if not math.isfinite(current_iv) or current_iv <= 0:
                return None
            # Compute IVR using the India VIX 52-week range (May 2026 research).
            # IV_52W_LOW / IV_52W_HIGH are env-var overridable; update quarterly.
            return calculate_iv_rank(current_iv, IV_52W_LOW, IV_52W_HIGH)
        except Exception as exc:
            print(f"[DATA] IV rank error: {exc}")
        return None

    def _fetch_target_expiry(self, symbol: str) -> str | None:
        """
        Select the nearest expiry for *symbol* that falls within the
        [DTE_MIN, DTE_MAX] window configured by environment variables.

        This implements the check_all_checkpoints expiry-selection checkpoint
        and fixes the audit finding that DTE_MIN/DTE_MAX were declared but
        never used — the strategy accepted whatever expiry the optionchain API
        returned without any DTE guard.

        Strategy:
          1. Ask ``client.expiry()`` for all available expiries.
          2. Parse each expiry string into a date and compute DTE.
          3. Return the first expiry inside the window, or None.
          4. Falls back to None when the expiry API is unavailable so the
             caller can gracefully skip the underlying.

        Returns an expiry string in the broker's native format (e.g. "30DEC25").
        """
        if not hasattr(self.client, "expiry"):
            return None   # SDK version does not expose expiry listing
        try:
            resp = self.client.expiry(
                symbol=symbol,
                exchange=FNO_EXCHANGE,
                instrumenttype="options",
            )
            if not resp:
                return None
            # SDK may return a list of strings or a dict with a "data" list
            expiry_list: list[str]
            if isinstance(resp, list):
                expiry_list = resp
            elif isinstance(resp, dict):
                expiry_list = resp.get("data", resp.get("expiries", []))
            else:
                return None

            now = datetime.now().date()
            for exp in expiry_list:
                exp_text = str(exp).strip().upper()
                exp_date = None
                for fmt in ("%d%b%y", "%d-%b-%y", "%d%b%Y", "%d-%b-%Y"):
                    try:
                        exp_date = datetime.strptime(exp_text, fmt).date()
                        break
                    except ValueError:
                        pass
                if exp_date is None:
                    continue
                dte = (exp_date - now).days
                if DTE_MIN <= dte <= DTE_MAX:
                    # expiry() docs return DD-MMM-YY; optionchain() expects DDMMMYY.
                    return exp_date.strftime("%d%b%y").upper()
            return None
        except Exception as exc:
            print(f"[DATA] expiry fetch error for {symbol}: {exc}")
            return None

    def _poll_order_status(
        self,
        order_id: str,
        max_retries: int = ORDER_STATUS_MAX_RETRIES,
        sleep_secs: float = ORDER_STATUS_POLL_INTERVAL,
    ) -> tuple[str | None, float | None, dict]:
        """
        Poll orderstatus and preserve pending state for accepted-but-open orders.
        Returns (status, average_price, raw_order_data).
        """
        if not order_id:
            return None, None, {}
        last_status = None
        last_data: dict = {}
        for attempt in range(max_retries):
            time.sleep(sleep_secs)
            try:
                resp = self.client.orderstatus(
                    order_id=order_id, strategy=self.strategy_name
                )
                od = resp.get("data", {}) if resp else {}
                last_data = od
                last_status = str(od.get("order_status", "")).lower() or None
                if last_status == "complete":
                    p = float(od.get("average_price", 0) or 0)
                    return "complete", p if p > 0 else None, od
                if last_status in ("rejected", "cancelled", "canceled"):
                    print(f"[ORDER] Order {order_id} {last_status} - no fill")
                    return last_status, None, od
            except Exception as exc:
                print(f"[ORDER] orderstatus poll error (attempt {attempt + 1}): {exc}")
        print(f"[ORDER] {order_id} still pending after {max_retries} retries - keeping for reconciliation")
        return last_status, None, last_data

    # ── Broker Order Helpers ─────────────────────────────────────────────────

    def _cancel_broker_orders(self, underlying: str) -> dict | None:
        """
        Cancel any pending broker-side SL-M and LIMIT target orders for *underlying*.
        Called before software-initiated exits so the broker orders don't double-execute.

        Returns True if any broker order was found to be already *complete* (filled at
        the exchange) so the caller can skip the software market-SELL and avoid
        an oversold / double-exit race.

        Strategy per order:
          1. Check current orderstatus first — if already complete, no cancel needed;
             set broker_filled=True so the caller skips the market SELL.
          2. Call cancelorder.
          3. Re-check orderstatus after a brief pause; if now complete, the SL-M or
             target filled *during* the cancel/market-sell window — treat as filled.
        """
        pos = self.positions.get(underlying)
        if not pos:
            return False
        broker_filled: dict | None = None
        for key, label in (("sl_order_id", "SL-M"), ("tgt_order_id", "Target LIMIT")):
            oid = pos.get(key)
            if not oid:
                continue
            # Step 1: pre-cancel status check
            try:
                pre_resp = self.client.orderstatus(order_id=oid, strategy=self.strategy_name)
                pre_od   = pre_resp.get("data", {}) if pre_resp else {}
                pre_status = str(pre_od.get("order_status", "")).lower()
                if pre_status == "complete":
                    print(f"[CANCEL] {underlying}: {label} order {oid} already filled — skip cancel")
                    pos[key] = None
                    broker_filled = {
                        "order_id": oid,
                        "label": label,
                        "price": float(pre_od.get("average_price", 0) or 0),
                    }
                    continue
            except Exception as exc:
                print(f"[CANCEL] {underlying}: {label} pre-cancel status error: {exc}")

            # Step 2: send cancel
            try:
                self.client.cancelorder(order_id=oid, strategy=self.strategy_name)
                print(f"[CANCEL] {underlying}: {label} order {oid} cancel sent")
            except Exception as exc:
                print(f"[CANCEL] {underlying}: {label} cancel failed (may already be done): {exc}")

            # Step 3: post-cancel status check — detect fill-during-cancel race
            time.sleep(0.3)
            try:
                post_resp = self.client.orderstatus(order_id=oid, strategy=self.strategy_name)
                post_od   = post_resp.get("data", {}) if post_resp else {}
                post_status = str(post_od.get("order_status", "")).lower()
                if post_status == "complete":
                    print(f"[CANCEL] {underlying}: {label} order {oid} filled during cancel window!")
                    pos[key] = None
                    broker_filled = {
                        "order_id": oid,
                        "label": label,
                        "price": float(post_od.get("average_price", 0) or 0),
                    }
                elif post_status in ("cancelled", "canceled", "rejected"):
                    pos[key] = None
                else:
                    print(f"[CANCEL] {underlying}: {label} order {oid} status unknown/open ({post_status}); keeping id")
            except Exception as exc:
                print(f"[CANCEL] {underlying}: {label} post-cancel status error: {exc}")

        return broker_filled

    def _modify_broker_sl(self, underlying: str, new_trigger: float):
        """
        Modify the broker-side SL-M order trigger to *new_trigger* when the trailing
        stop-loss ratchets upward for the option position on *underlying*.
        Does nothing when BROKER_SL_ORDERS is disabled or no SL order ID is recorded.
        """
        if not BROKER_SL_ORDERS:
            return
        pos = self.positions.get(underlying)
        if not pos or not pos.get("sl_order_id"):
            return
        try:
            resp = self.client.modifyorder(
                order_id=pos["sl_order_id"],
                strategy=self.strategy_name,
                symbol=pos["symbol"],
                action="SELL",
                exchange=FNO_EXCHANGE,
                price_type="SL-M",
                product="NRML",
                quantity=pos["qty"],
                price=0,
                trigger_price=new_trigger,
            )
            if resp and resp.get("status") == "success":
                print(f"[BROKER-SL] {underlying}: SL-M trigger raised to ₹{new_trigger:.2f}")
            else:
                print(f"[BROKER-SL] {underlying}: SL modify failed: {resp}")
        except Exception as exc:
            print(f"[BROKER-SL] {underlying}: SL modify exception: {exc}")

    def _check_broker_order_fills(self):
        """
        Periodically query broker order status for each tracked position to detect
        exchange-level SL or target fills that occurred without script involvement
        (e.g. while WebSocket was momentarily disconnected, or after a fast price gap).

        When a fill is detected:
          • The complementary order (target ↔ SL) is cancelled.
          • P&L is updated and the position state is fully cleaned up.
        """
        for ul in list(self.positions.keys()):
            pos = self.positions.get(ul)
            if not pos:
                continue
            opt_sym = pos["symbol"]
            with self.exit_lock:
                if opt_sym in self.exit_queue:
                    continue   # software exit already in progress

            for key, label in (("sl_order_id", "SL"), ("tgt_order_id", "Target")):
                oid = pos.get(key)
                if not oid:
                    continue
                try:
                    resp = self.client.orderstatus(order_id=oid, strategy=self.strategy_name)
                    od = resp.get("data", {}) if resp else {}
                    status = od.get("order_status", "")
                    if status != "complete":
                        continue
                except Exception as exc:
                    print(f"[BROKER-CHECK] orderstatus error for {ul} {label} order {oid}: {exc}")
                    continue

                # Broker order filled — clean up position
                with self.exit_lock:
                    if opt_sym in self.exit_queue:
                        break   # another exit already running
                    self.exit_queue.add(opt_sym)

                exit_price = (
                    float(od.get("average_price", 0) or 0)
                    or self.ltp_map.get(opt_sym, pos["entry_premium"])
                )
                pnl      = (exit_price - pos["entry_premium"]) * pos["qty"]
                pnl_sign = "✅" if pnl >= 0 else "❌"
                reason   = f"BROKER {label.upper()} FILLED @ ₹{exit_price:.2f}"
                print(f"\n[BROKER-FILL] {ul} {pos.get('option_type','')}: {reason} | P&L ₹{pnl:.2f}")

                # Update session / daily risk state
                with self.state_lock:
                    self.daily_pnl += pnl
                    if pnl < 0:
                        self.session_consecutive_losses += 1
                        print(f"[RISK] Loss streak: {self.session_consecutive_losses} | "
                              f"Daily P&L ₹{self.daily_pnl:.0f}")
                    else:
                        self.session_consecutive_losses = 0

                self._send_telegram(
                    f"{pnl_sign} {self.strategy_name} EXIT (broker)\n"
                    f"{ul} {pos.get('option_type','')} | {opt_sym}\n"
                    f"Reason: {reason}\n"
                    f"Exit ₹{exit_price:.2f} | Entry ₹{pos['entry_premium']:.2f} | P&L ₹{pnl:.2f}\n"
                    f"Daily P&L ₹{self.daily_pnl:.0f}",
                    priority=8 if pnl < 0 else 6,
                )

                # Cancel the complementary open order
                other_key = "tgt_order_id" if key == "sl_order_id" else "sl_order_id"
                other_oid = pos.get(other_key)
                if other_oid:
                    try:
                        self.client.cancelorder(order_id=other_oid, strategy=self.strategy_name)
                        print(f"[CANCEL] {ul}: complementary order {other_oid} cancelled")
                    except Exception as exc:
                        print(f"[CANCEL] {ul}: complementary cancel failed: {exc}")

                # Unsubscribe WebSocket feeds and remove position.
                # state_lock must be held when mutating positions to avoid a race
                # with _strategy_thread / _ws_thread reading the positions dict.
                self._unsubscribe(FNO_EXCHANGE, opt_sym)
                if TRAIL_SL_MODE in ("spot", "both"):
                    self._unsubscribe_spot(pos.get("spot_symbol", ul))
                with self.state_lock:
                    self.positions.pop(ul, None)
                    self._prev_straddle.pop(ul, None)
                with self.exit_lock:
                    self.exit_queue.discard(opt_sym)

                break   # position cleaned up; no need to check the other order

    # ── Order Placement ──────────────────────────────────────────────────────

    def _register_filled_entry(
        self,
        underlying: str,
        option_symbol: str,
        qty: int,
        spot: float,
        direction: str,
        executed: float,
    ) -> bool:
        """Record a confirmed BUY fill and place broker-side protection."""
        sl = round(executed * (1 - PREMIUM_STOP_PCT / 100), 2)
        tgt = round(executed * (1 + PREMIUM_TARGET_PCT / 100), 2)
        reward_dist = spot * SPOT_REWARD_PCT / 100.0

        with self.state_lock:
            self.positions[underlying] = {
                "symbol": option_symbol,
                "entry_premium": executed,
                "qty": qty,
                "option_type": direction,
                "sl": sl,
                "tgt": tgt,
                "spot_symbol": underlying,
                "spot_entry": spot,
                "reward_dist": reward_dist,
                "trail_active": False,
                "trail_peak": None,
                "trail_sl_spot": None,
                "premium_trail_active": False,
                "premium_trail_peak": None,
                "premium_trail_sl": None,
                "sl_order_id": None,
                "tgt_order_id": None,
                "broker_protection": False,
            }

        sl_placed = False
        tgt_placed = False
        if BROKER_SL_ORDERS:
            for attempt in range(3):
                try:
                    sl_resp = self.client.placeorder(
                        strategy=self.strategy_name,
                        symbol=option_symbol,
                        exchange=FNO_EXCHANGE,
                        action="SELL",
                        quantity=qty,
                        price_type="SL-M",
                        product="NRML",
                        price=0,
                        trigger_price=sl,
                    )
                    if sl_resp and sl_resp.get("status") == "success":
                        with self.state_lock:
                            self.positions[underlying]["sl_order_id"] = sl_resp.get("orderid")
                        sl_placed = True
                        print(
                            f"        [BROKER-SL]  SL-M order placed  | "
                            f"trigger=₹{sl:.2f} | id={sl_resp.get('orderid')}"
                        )
                        break
                    print(f"[WARN] Broker SL-M attempt {attempt + 1} failed: {sl_resp}")
                    time.sleep(1)
                except Exception as exc:
                    print(f"[WARN] Broker SL-M attempt {attempt + 1} exception: {exc}")
                    time.sleep(1)

            if not sl_placed:
                self._send_telegram(
                    f"🚨 {self.strategy_name}: BROKER SL-M FAILED for "
                    f"{underlying} {direction} {option_symbol}\n"
                    f"Entry ₹{executed:.2f} | SL should be ₹{sl:.2f}\n"
                    "⚠️ POSITION UNPROTECTED — place SL-M manually!",
                    priority=9,
                )
            else:
                try:
                    tgt_resp = self.client.placeorder(
                        strategy=self.strategy_name,
                        symbol=option_symbol,
                        exchange=FNO_EXCHANGE,
                        action="SELL",
                        quantity=qty,
                        price_type="LIMIT",
                        product="NRML",
                        price=tgt,
                    )
                    if tgt_resp and tgt_resp.get("status") == "success":
                        with self.state_lock:
                            self.positions[underlying]["tgt_order_id"] = tgt_resp.get("orderid")
                        tgt_placed = True
                        print(
                            f"        [BROKER-TGT] LIMIT order placed  | "
                            f"price=₹{tgt:.2f}   | id={tgt_resp.get('orderid')}"
                        )
                    else:
                        print(f"[WARN] Broker target LIMIT order not placed: {tgt_resp}")
                except Exception as exc:
                    print(f"[WARN] Broker target order exception: {exc}")

                if tgt_placed:
                    with self.state_lock:
                        self.positions[underlying]["broker_protection"] = True

        self._subscribe(FNO_EXCHANGE, option_symbol)
        if TRAIL_SL_MODE in ("spot", "both"):
            self._subscribe_spot(underlying)

        with self.state_lock:
            self.session_trade_count += 1
            self.last_entry_time = datetime.now()
            broker_ok = self.positions[underlying].get("broker_protection")

        print(f"[ENTRY] {underlying} | {option_symbol} × {qty} @ ₹{executed:.2f}")
        print(f"        SL ₹{sl:.2f} | TGT ₹{tgt:.2f} | "
              f"Broker protection: {'OK' if broker_ok else 'PARTIAL/NONE'}")
        if TRAIL_SL_MODE in ("spot", "both"):
            print(f"        Spot entry {spot:.1f} | reward dist {reward_dist:.1f} pts | "
                  f"spot trail activates at +{reward_dist * TRAIL_ACTIVATE_AT_PCT / 100:.1f} pts")
        if TRAIL_SL_MODE in ("premium", "both"):
            prem_reward = executed * (PREMIUM_TARGET_PCT / 100.0)
            print(f"        Premium entry ₹{executed:.2f} | prem reward ₹{prem_reward:.2f} | "
                  f"trail activates at +₹{prem_reward * TRAIL_ACTIVATE_AT_PCT / 100:.2f}")

        self._send_telegram(
            f"📈 {self.strategy_name} ENTRY\n"
            f"{underlying} {direction} | {option_symbol} × {qty}\n"
            f"Entry ₹{executed:.2f} | SL ₹{sl:.2f} | TGT ₹{tgt:.2f}\n"
            f"Spot ₹{spot:.1f}",
            priority=7,
        )
        return True

    def _place_entry(
        self,
        underlying: str,
        option_symbol: str,
        qty: int,
        spot: float,
        direction: str,
    ) -> bool:
        """
        Place a BUY order and record the position with all SL, target and
        trailing-SL state fields.

        Parameters
        ----------
        underlying    : underlying name (e.g. "NIFTY")
        option_symbol : full option trading symbol
        qty           : number of shares/units to buy
        spot          : current spot LTP at entry (used for spot-based trailing SL)
        direction     : "CE" or "PE"
        """
        print(f"[ORDER] BUY {qty} × {option_symbol}")
        try:
            resp = self.client.placeorder(
                strategy=self.strategy_name,
                symbol=option_symbol,
                exchange=FNO_EXCHANGE,
                action="BUY",
                quantity=qty,
                price_type="MARKET",
                product="NRML",
            )
            if resp.get("status") == "success":
                order_id = resp.get("orderid")
                if not order_id:
                    print(f"[ERROR] BUY accepted without order id: {resp}")
                    self._send_telegram(
                        f"🚨 {self.strategy_name}: BUY response missing order id for "
                        f"{underlying} {direction} {option_symbol}. Verify orderbook manually.",
                        priority=9,
                    )
                    return False
                status, executed, _ = self._poll_order_status(order_id)
                if not executed:
                    if status in ("rejected", "cancelled", "canceled"):
                        print(f"[ERROR] BUY order {order_id} {status}: {resp}")
                        return False
                    if status not in ("rejected", "cancelled", "canceled"):
                        with self.state_lock:
                            self.pending_entries[underlying] = {
                                "order_id": order_id,
                                "symbol": option_symbol,
                                "qty": qty,
                                "spot": spot,
                                "direction": direction,
                                "created_at": datetime.now(),
                            }
                    # Order accepted but still pending after retries.
                    # Keep tracking via order_id so we don't abandon a live position.
                    print(
                        f"[WARN] BUY order {order_id} pending — tracking for reconciliation. "
                        "Manual review required."
                    )
                    self._send_telegram(
                        f"⚠️ {self.strategy_name}: BUY order {order_id} for "
                        f"{underlying} {direction} is pending after retries.\n"
                        "Monitor via OpenAlgo orderbook and update manually.",
                        priority=9,
                    )
                    return False

                return self._register_filled_entry(
                    underlying, option_symbol, qty, spot, direction, executed
                )
            print(f"[ERROR] Order failed: {resp}")
        except Exception as exc:
            print(f"[ERROR] Entry order exception: {exc}")
        return False

    def _place_exit(self, underlying: str, reason: str):
        pos = self.positions.get(underlying)
        if not pos:
            with self.exit_lock:
                self.exit_queue.discard("")
            return
        opt_sym = pos["symbol"]
        print(f"[EXIT] Closing {underlying} {opt_sym} — {reason}")

        # Cancel pending broker-side SL-M and target orders before placing the
        # software exit so the exchange does not execute both sides.
        # Returns True if a broker order was already filled (double-exit risk).
        broker_fill = self._cancel_broker_orders(underlying)

        if broker_fill:
            # A broker SL-M or target order already filled while we tried to cancel.
            # Do NOT place another market SELL — the position is already squared off.
            # Use last known LTP as proxy exit price for P&L reporting.
            print(f"[EXIT] {underlying}: broker order already filled — skipping software SELL")
            exit_price = broker_fill.get("price") or self.ltp_map.get(opt_sym, pos["entry_premium"])
            pnl = (exit_price - pos["entry_premium"]) * pos["qty"]
            pnl_sign = "✅" if pnl >= 0 else "❌"
            with self.state_lock:
                self.daily_pnl += pnl
                if pnl < 0:
                    self.session_consecutive_losses += 1
                else:
                    self.session_consecutive_losses = 0
            self._send_telegram(
                f"{pnl_sign} {self.strategy_name} EXIT (broker-filled, no sw-SELL)\n"
                f"{underlying} {pos.get('option_type','')} | {opt_sym}\n"
                f"Reason: {reason}\n"
                f"Est. exit ₹{exit_price:.2f} | Entry ₹{pos['entry_premium']:.2f} | P&L ≈₹{pnl:.2f}\n"
                f"Daily P&L ₹{self.daily_pnl:.0f}",
                priority=8 if pnl < 0 else 6,
            )
            self._unsubscribe(FNO_EXCHANGE, opt_sym)
            if TRAIL_SL_MODE in ("spot", "both"):
                self._unsubscribe_spot(pos.get("spot_symbol", underlying))
            with self.state_lock:
                self.positions.pop(underlying, None)
                self._prev_straddle.pop(underlying, None)
            with self.exit_lock:
                self.exit_queue.discard(opt_sym)
            return

        # Standard software-initiated exit: place market SELL and confirm fill.
        exit_confirmed = False
        try:
            resp = self.client.placeorder(
                strategy=self.strategy_name,
                symbol=opt_sym,
                exchange=FNO_EXCHANGE,
                action="SELL",
                quantity=pos["qty"],
                price_type="MARKET",
                product="NRML",
            )
            if resp.get("status") == "success":
                exit_order_id = resp.get("orderid")
                if not exit_order_id:
                    print(f"[EXIT ERROR] SELL accepted without order id: {resp}")
                    self._send_telegram(
                        f"🚨 {self.strategy_name}: EXIT response missing order id for {underlying} {opt_sym}\n"
                        "Verify orderbook manually before retrying.",
                        priority=9,
                    )
                    with self.exit_lock:
                        self.exit_queue.discard(opt_sym)
                    return
                # Confirm fill via orderstatus.  A pending SELL must stay tracked;
                # falling back to LTP here would hide a live position.
                status, actual_exit, _ = self._poll_order_status(exit_order_id) if exit_order_id else (None, None, {})
                if status in ("rejected", "cancelled", "canceled"):
                    print(f"[EXIT ERROR] SELL order {exit_order_id} {status}")
                    self._send_telegram(
                        f"🚨 {self.strategy_name}: EXIT FAILED for {underlying} {opt_sym}\n"
                        f"SELL order {exit_order_id} {status}\n"
                        "⚠️ POSITION STILL OPEN — manual exit required!",
                        priority=9,
                    )
                    with self.exit_lock:
                        self.exit_queue.discard(opt_sym)
                    return
                if status != "complete" or actual_exit is None:
                    with self.state_lock:
                        self.pending_exits[underlying] = {
                            "order_id": exit_order_id,
                            "reason": reason,
                            "created_at": datetime.now(),
                        }
                        pos["exit_pending"] = True
                    print(f"[EXIT] {underlying}: SELL order {exit_order_id} pending — keeping position tracked")
                    self._send_telegram(
                        f"⚠️ {self.strategy_name}: EXIT order pending\n"
                        f"{underlying} {opt_sym} | order {exit_order_id}\n"
                        "Position remains tracked until orderstatus confirms completion.",
                        priority=8,
                    )
                    return
                exit_price = actual_exit
                exit_confirmed = True

                pnl = (exit_price - pos["entry_premium"]) * pos["qty"]
                pnl_sign = "✅" if pnl >= 0 else "❌"
                print(f"[EXIT] {underlying}: exit ₹{exit_price:.2f} | P&L ₹{pnl:.2f}")

                # Update session / daily risk state
                with self.state_lock:
                    self.daily_pnl += pnl
                    if pnl < 0:
                        self.session_consecutive_losses += 1
                        print(f"[RISK] Loss streak: {self.session_consecutive_losses} | "
                              f"Daily P&L ₹{self.daily_pnl:.0f}")
                    else:
                        self.session_consecutive_losses = 0   # reset on a win

                self._send_telegram(
                    f"{pnl_sign} {self.strategy_name} EXIT\n"
                    f"{underlying} {pos.get('option_type','')} | {opt_sym}\n"
                    f"Reason: {reason}\n"
                    f"Exit ₹{exit_price:.2f} | Entry ₹{pos['entry_premium']:.2f} | P&L ₹{pnl:.2f}\n"
                    f"Daily P&L ₹{self.daily_pnl:.0f}",
                    priority=8 if pnl < 0 else 6,
                )
            else:
                # placeorder returned a non-success status — keep the position tracked.
                print(f"[EXIT ERROR] SELL order rejected: {resp}")
                self._send_telegram(
                    f"🚨 {self.strategy_name}: EXIT FAILED for {underlying} {opt_sym}\n"
                    f"SELL order rejected: {resp}\n"
                    "⚠️ POSITION STILL OPEN — manual exit required!",
                    priority=9,
                )
        except Exception as exc:
            # Connectivity or unexpected error — keep tracking the position.
            print(f"[EXIT ERROR] Exception during SELL: {exc}")
            self._send_telegram(
                f"🚨 {self.strategy_name}: EXIT ERROR for {underlying} {opt_sym}\n"
                f"Exception: {exc}\n"
                "⚠️ POSITION MAY STILL BE OPEN — verify and exit manually!",
                priority=9,
            )

        if exit_confirmed:
            # Only clean up after a confirmed exit to avoid losing track of live positions.
            self._unsubscribe(FNO_EXCHANGE, opt_sym)
            if TRAIL_SL_MODE in ("spot", "both"):
                spot_sym = pos.get("spot_symbol", underlying) if pos else underlying
                self._unsubscribe_spot(spot_sym)
            with self.state_lock:
                self.positions.pop(underlying, None)
            # Clear the straddle cache so the next entry on this underlying
            # starts with a fresh prev_straddle_price rather than a stale value
            # from the previous position's holding period.
            self._prev_straddle.pop(underlying, None)
            with self.exit_lock:
                self.exit_queue.discard(opt_sym)
        else:
            # Exit was NOT confirmed — clear the exit_queue flag so the software
            # trail / SL logic can re-trigger an exit attempt on the next tick.
            with self.exit_lock:
                self.exit_queue.discard(opt_sym)

    # ── Signal Loop ──────────────────────────────────────────────────────────

    def _scan_underlying(self, symbol: str):
        with self.state_lock:
            already_active = symbol in self.positions or symbol in self.pending_entries
        if already_active:
            return   # already in a trade for this underlying

        # Check all session-level risk gates before doing any expensive fetches
        allowed, gate_reason = self._check_risk_gates()
        if not allowed:
            print(f"[RISK GATE] {symbol}: {gate_reason} — skipping")
            return

        print(f"\n[SCAN] {symbol}")

        # --- gather data ---
        df_spot = self._fetch_spot_candles(symbol)
        spot_q  = self._fetch_quote(symbol, self._underlying_exchange(symbol))
        spot    = float(spot_q.get("ltp", 0) or 0)
        if not spot:
            print(f"[SCAN] {symbol}: no spot LTP, skipping")
            return

        # Select the target expiry within the DTE_MIN–DTE_MAX window.
        # This ensures we never enter a weekly expiry or a contract outside the
        # intended DTE range.  `_fetch_target_expiry` handles the DTE filter.
        target_expiry = self._fetch_target_expiry(symbol)
        if not target_expiry:
            print(f"[SCAN] {symbol}: no expiry in DTE_MIN={DTE_MIN}–DTE_MAX={DTE_MAX} range, skipping")
            return

        chain_rows, chain_expiry = self._fetch_option_chain(symbol, expiry=target_expiry)
        if not chain_rows:
            print(f"[SCAN] {symbol}: empty option chain, skipping")
            return

        # SMA-smooth chain OI/Vol/Premium over LOOKBACK_BARS bars for L2 signals.
        # Raw chain_rows kept for ATM extraction, straddle velocity, and delta.
        _hist = self._chain_history.setdefault(symbol, deque(maxlen=max(1, LOOKBACK_BARS)))
        _hist.append(chain_rows)
        chain_rows_smooth = _smooth_chain_rows(list(_hist))

        # Use the expiry returned by the chain API (may contain canonical formatting).
        # Fall back to the pre-selected target_expiry if the API doesn't echo it back.
        expiry_used = chain_expiry or target_expiry

        # ATM CE and PE LTP and bid/ask (for Greeks proxy and spread checks)
        strikes = sorted(set(r.get("strike", 0) for r in chain_rows if r.get("strike")))
        atm = min(strikes, key=lambda x: abs(x - spot)) if strikes else None
        atm_ce_ltp = atm_pe_ltp = None
        atm_ce_symbol = atm_pe_symbol = None
        ce_bid = ce_ask = pe_bid = pe_ask = None
        if atm:
            for row in chain_rows:
                if row.get("strike") == atm:
                    atm_ce_ltp    = row.get("ce_ltp")
                    atm_pe_ltp    = row.get("pe_ltp")
                    atm_ce_symbol = row.get("ce_symbol")
                    atm_pe_symbol = row.get("pe_symbol")
                    ce_bid = row.get("ce_bid")
                    ce_ask = row.get("ce_ask")
                    pe_bid = row.get("pe_bid")
                    pe_ask = row.get("pe_ask")
                    break

        # Straddle price (ATM CE + ATM PE)
        straddle_price = (
            (atm_ce_ltp + atm_pe_ltp)
            if atm_ce_ltp is not None and atm_pe_ltp is not None
            else None
        )
        
        # Institutional Straddle Velocity Audit:
        # Only compare premium expansion if we are looking at the SAME strike as the previous minute.
        # If the ATM strike shifted, velocity is undefined/reset for this bar.
        prev_data = self._prev_straddle.get(symbol, {})
        prev_straddle_price = None
        if isinstance(prev_data, dict) and prev_data.get("strike") == atm:
            prev_straddle_price = prev_data.get("price")
            
        if straddle_price is not None:
            self._prev_straddle[symbol] = {"strike": atm, "price": straddle_price}

        # Synthetic future price via client.syntheticfuture() for index underlyings;
        # falls back to a near-month futures quote for equity underlyings.
        # Pass the selected expiry so the SF price matches the chain expiry.
        sf_ltp = self._fetch_synthetic_future(symbol, expiry_used)

        # Previous scan's spot and SF prices for co-movement scoring (component 13).
        # These are updated AFTER scoring so consecutive scans compare correctly.
        prev_spot  = self._prev_spot.get(symbol)
        prev_sf    = self._prev_sf.get(symbol)

        # Actual ATM option greeks from client.optiongreeks() — provides the real
        # delta skew instead of estimating from LTP ratios.
        ce_delta, pe_delta = self._fetch_atm_greeks(symbol, atm_ce_symbol, atm_pe_symbol)

        # IV Rank — derived from the already-fetched spot_q (no extra API call).
        iv_rank = self._fetch_iv_rank(spot_q)
        if iv_rank is not None and iv_rank >= IV_RANK_MAX_ENTRY:
            print(f"[SKIP] {symbol}: IVR {iv_rank:.1f}% >= max {IV_RANK_MAX_ENTRY:.1f}%")
            return

        # --- compute score ---
        result = compute_composite_score(
            spot                = spot,
            df_spot             = df_spot,
            chain_rows          = chain_rows_smooth,
            atm_ce_ltp          = atm_ce_ltp,
            atm_pe_ltp          = atm_pe_ltp,
            iv_rank             = iv_rank,
            straddle_price      = straddle_price,
            prev_straddle_price = prev_straddle_price,
            sf_ltp              = sf_ltp,
            ce_bid              = ce_bid,
            ce_ask              = ce_ask,
            pe_bid              = pe_bid,
            pe_ask              = pe_ask,
            ce_delta            = ce_delta,
            pe_delta            = pe_delta,
            prev_spot           = prev_spot,
            prev_sf_ltp         = prev_sf,
        )

        # Update previous spot and SF snapshots for the next scan cycle.
        self._prev_spot[symbol] = spot
        if sf_ltp is not None:
            self._prev_sf[symbol] = sf_ltp

        score      = result["score"]
        signal     = result["signal"]
        label      = result["label"]
        trap_score = result["trap_score"]
        direction  = result["direction"]

        print(f"[SCORE] {symbol}: {score:+d} ({label}) | trap={trap_score} | signal={signal}")
        for note in result["reasons"][:4]:
            print(f"        • {note}")

        if signal != "EXECUTE":
            print(f"[SKIP] {symbol}: signal={signal}, not executing")
            return

        # Long-Options mode: Option buyers go long on momentum.
        # CE buys for upward momentum, PE buys for downward momentum.
        # Both are options-buyer (long) trades — no short-selling of options here.
        if not LONG_ONLY_MODE:
            print(f"[SKIP] {symbol}: Strategy is long-options, but LONG_ONLY_MODE is disabled.")
            return

        # direction=None means truly neutral score — skip rather than default to PE.
        if direction is None:
            print(f"[SKIP] {symbol}: direction=None (neutral score) — skipping")
            return

        # --- select strike (check_all_checkpoints best-strike combined with OTM fallback) ---
        # Step 1: try the check_all_checkpoints liquidity + asymmetry score filter.
        #   This selects the OTM strike with the best institutional-edge profile
        #   (OI concentration, IV regime, delta range).
        opt_row = self._select_best_strike(symbol, chain_rows_smooth, spot, direction, iv_rank)

        # Step 2: if no strike passes the checkpoint criteria, fall back to the
        #   OTM_OFFSET-based simple selection so the strategy can still enter when
        #   the chain is thinly populated or the asymmetry threshold is marginal.
        if opt_row is None:
            if not ALLOW_CHECKPOINT_FALLBACK:
                print(f"[SKIP] {symbol}: checkpoint strike criteria not met")
                return
            print(f"[STRIKE] {symbol}: checkpoint criteria not met — falling back to OTM offset={OTM_OFFSET}")
            opt_row = select_option_strike(chain_rows, spot, direction, OTM_OFFSET)

        if not opt_row:
            print(f"[SKIP] {symbol}: could not select any strike")
            return

        target_strike = opt_row.get("strike")
        option_ltp    = float(opt_row.get(f"{direction.lower()}_ltp") or opt_row.get("ltp", 0))
        if not option_ltp:
            print(f"[SKIP] {symbol}: option LTP is 0")
            return

        # Build option symbol from the flattened chain row (ce_symbol / pe_symbol keys).
        opt_symbol = opt_row.get(f"{direction.lower()}_symbol") or opt_row.get("symbol")
        if not opt_symbol:
            print(f"[SKIP] {symbol}: option symbol not in chain row")
            return

        lotsize = int(opt_row.get("lotsize", 1) or 1)
        fixed_qty = LOT_MULTIPLIER * lotsize
        available_capital = self._available_capital()
        risk_cap = available_capital * (RISK_PERCENT / 100.0)
        risk_per_unit = option_ltp * (PREMIUM_STOP_PCT / 100.0)
        risk_qty = int(risk_cap / risk_per_unit) if risk_per_unit > 0 else 0
        risk_qty = (risk_qty // lotsize) * lotsize if lotsize > 0 else risk_qty
        qty = min(fixed_qty, risk_qty) if risk_qty > 0 else 0
        if qty <= 0:
            print(
                f"[SKIP] {symbol}: 1 lot risk exceeds cap "
                f"(premium ₹{option_ltp:.2f}, stop {PREMIUM_STOP_PCT}%, "
                f"risk cap ₹{risk_cap:.0f})"
            )
            return

        print(f"[SIGNAL] {symbol}: BUY {direction} | strike={target_strike} | premium=₹{option_ltp:.2f} | qty={qty}")

        self._place_entry(symbol, opt_symbol, qty, spot=spot, direction=direction)

    def _check_pending_entries(self):
        with self.state_lock:
            pending = list(self.pending_entries.items())
        for ul, info in pending:
            order_id = info.get("order_id")
            status, price, _ = self._poll_order_status(order_id, max_retries=1, sleep_secs=0)
            if status == "complete" and price:
                with self.state_lock:
                    self.pending_entries.pop(ul, None)
                    already_open = ul in self.positions
                if already_open:
                    self._send_telegram(
                        f"⚠️ {self.strategy_name}: pending BUY {order_id} filled but "
                        f"{ul} already has a tracked position. Reconcile manually.",
                        priority=9,
                    )
                    continue
                print(f"[PENDING] BUY {order_id} filled for {ul} @ ₹{price:.2f}; activating protection")
                self._register_filled_entry(
                    ul,
                    info["symbol"],
                    int(info["qty"]),
                    float(info["spot"]),
                    info["direction"],
                    price,
                )
            elif status in ("rejected", "cancelled", "canceled"):
                with self.state_lock:
                    self.pending_entries.pop(ul, None)
                print(f"[PENDING] BUY {order_id} {status}; removed from pending entries")

    def _check_pending_exits(self):
        with self.state_lock:
            pending = list(self.pending_exits.items())
        for ul, info in pending:
            order_id = info.get("order_id")
            status, price, _ = self._poll_order_status(order_id, max_retries=1, sleep_secs=0)
            with self.state_lock:
                pos = self.positions.get(ul)
            if not pos:
                with self.state_lock:
                    self.pending_exits.pop(ul, None)
                continue
            opt_sym = pos["symbol"]
            if status == "complete" and price:
                pnl = (price - pos["entry_premium"]) * pos["qty"]
                pnl_sign = "✅" if pnl >= 0 else "❌"
                with self.state_lock:
                    self.daily_pnl += pnl
                    if pnl < 0:
                        self.session_consecutive_losses += 1
                    else:
                        self.session_consecutive_losses = 0
                    self.positions.pop(ul, None)
                    self.pending_exits.pop(ul, None)
                    self._prev_straddle.pop(ul, None)
                self._unsubscribe(FNO_EXCHANGE, opt_sym)
                if TRAIL_SL_MODE in ("spot", "both"):
                    self._unsubscribe_spot(pos.get("spot_symbol", ul))
                with self.exit_lock:
                    self.exit_queue.discard(opt_sym)
                print(f"[PENDING] EXIT {order_id} complete for {ul} @ ₹{price:.2f} | P&L ₹{pnl:.2f}")
                self._send_telegram(
                    f"{pnl_sign} {self.strategy_name} EXIT confirmed\n"
                    f"{ul} {pos.get('option_type','')} | {opt_sym}\n"
                    f"Exit ₹{price:.2f} | Entry ₹{pos['entry_premium']:.2f} | P&L ₹{pnl:.2f}\n"
                    f"Daily P&L ₹{self.daily_pnl:.0f}",
                    priority=8 if pnl < 0 else 6,
                )
            elif status in ("rejected", "cancelled", "canceled"):
                with self.state_lock:
                    self.pending_exits.pop(ul, None)
                    pos["exit_pending"] = False
                with self.exit_lock:
                    self.exit_queue.discard(opt_sym)
                self._send_telegram(
                    f"🚨 {self.strategy_name}: pending EXIT {order_id} {status} for {ul} {opt_sym}\n"
                    "Position remains tracked; software exit may retry on next trigger.",
                    priority=9,
                )

    def _strategy_thread(self):
        print("[STRATEGY] Thread started — scanning every "
              f"{SIGNAL_CHECK_INTERVAL}s")
        while not self.stop_event.is_set():
            try:
                self._check_pending_entries()
                self._check_pending_exits()
                for ul in UNDERLYINGS:
                    if self.stop_event.is_set():
                        break
                    self._scan_underlying(ul)

                # After scanning all underlyings, check whether any broker-side
                # SL or target orders have been filled at the exchange level
                # (protects against missed WebSocket ticks or script downtime).
                if BROKER_SL_ORDERS:
                    self._check_broker_order_fills()
            except Exception as exc:
                print(f"[STRATEGY ERROR] {exc}")

            # Clock-anchored sync: sleep to the next multiple of SIGNAL_CHECK_INTERVAL
            # from the epoch so scans align to regular boundaries (e.g., every minute
            # at 09:15:00, 09:16:00, … rather than drifting based on scan duration).
            # SIGNAL_CHECK_INTERVAL=60 → standard 1-minute bar alignment.
            # SIGNAL_CHECK_INTERVAL=30 → 30-second polling (2 scans per minute bar).
            interval = max(SIGNAL_CHECK_INTERVAL, 1)
            now = time.time()
            sleep_secs = interval - (now % interval)
            if sleep_secs < 0.5:
                sleep_secs += interval
            time.sleep(sleep_secs)

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 70)
        print(" OPTIONS MOMENTUM STRATEGY — Multi-Layer Confirmation")
        print("=" * 70)
        print(f" Underlyings : {', '.join(UNDERLYINGS)}")
        print(f" Spot exch   : {SPOT_EXCHANGE}  |  F&O exch: {FNO_EXCHANGE}")
        print(f" DTE         : {DTE_MIN}–{DTE_MAX} days  |  OTM offset: {OTM_OFFSET}")
        print(f" Score gate  : ≥{MIN_SCORE}  |  Trap gate: ≤{MAX_TRAP}")
        print(f" Premium SL  : {PREMIUM_STOP_PCT}%  |  Target: {PREMIUM_TARGET_PCT}%")
        print(f" Candle      : {CANDLE_INTERVAL}  |  Lookback: {LOOKBACK_DAYS}d")
        print(f" Loop        : every {SIGNAL_CHECK_INTERVAL}s")
        print("─" * 70)
        print(f" [RISK GATES]")
        print(f"  Max trades/session : {MAX_TRADES_PER_SESSION or 'unlimited'}")
        print(f"  Max loss streak    : {MAX_CONSECUTIVE_LOSSES or 'unlimited'}")
        print(f"  Entry cooldown     : {ENTRY_COOLDOWN_SECS}s")
        print(f"  Daily loss limit   : {MAX_DAILY_LOSS_PCT}% of live available cash | ₹{MAX_DAILY_LOSS_AMOUNT:.0f} absolute")
        print(f" [TRAILING SL — MODE: {TRAIL_SL_MODE.upper()}]")
        if TRAIL_SL_MODE in ("spot", "both"):
            print(f"  [Spot]    Reward target  : {SPOT_REWARD_PCT}% spot move")
        if TRAIL_SL_MODE in ("premium", "both"):
            print(f"  [Premium] Reward target  : {PREMIUM_TARGET_PCT}% of entry premium")
        print(f"  Activates after    : {TRAIL_ACTIVATE_AT_PCT}% of reward")
        print(f"  Trail step         : {TRAIL_STEP_RR_PCT}% of reward distance")
        print(f" [ORDER PROTECTION]")
        print(f"  Long-only mode     : {'ENABLED — CE (calls) only' if LONG_ONLY_MODE else 'disabled (CE + PE)'}")
        if BROKER_SL_ORDERS:
            print(f"  Broker SL orders   : ENABLED — SL-M @ -{PREMIUM_STOP_PCT}% + LIMIT @ +{PREMIUM_TARGET_PCT}%")
            print(f"                       Trailing SL modifies broker SL-M trigger as trail ratchets")
        else:
            print(f"  Broker SL orders   : disabled (software WebSocket monitoring only)")
        print("=" * 70)
        print("Press Ctrl+C to stop\n")

        ws_t = threading.Thread(target=self._ws_thread, daemon=True)
        ws_t.start()
        time.sleep(2)   # let WS connect

        st_t = threading.Thread(target=self._strategy_thread, daemon=True)
        st_t.start()

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Stopping bot...")
            self.running = False
            self.stop_event.set()

            # Close all open positions
            for ul in list(self.positions.keys()):
                print(f"[SHUTDOWN] Closing {ul} position...")
                self._place_exit(ul, "Bot Shutdown")

            ws_t.join(timeout=5)
            st_t.join(timeout=5)
            print("[DONE] Bot stopped.")


# ===============================================================================
# ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    _validate_config()

    if not API_KEY or API_KEY == "openalgo-apikey":
        print(
            "[WARNING] OPENALGO_API_KEY is not set in environment.\n"
            "          Export it before running: export OPENALGO_API_KEY=your-key"
        )

    print("=" * 70)
    print(" OPTIONS MOMENTUM STRATEGY — READY")
    print("=" * 70)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    bot = OptionsMomentumBot()
    bot.run()
