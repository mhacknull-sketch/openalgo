"""
Checkpoint-only prototype for long options selection with the OpenAlgo SDK.

This file intentionally does NOT place orders.  It is meant to validate filters
and select a candidate option row; execution should happen through the safer
order-management path in options_momentum_strategy.py.
"""

import os
from datetime import datetime

import pandas as pd
from openalgo import api

# ========================= CONFIG =========================
API_KEY = os.getenv("OPENALGO_API_KEY", "openalgo-apikey")
API_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
SPOT_EXCHANGE = os.getenv("OPENALGO_STRATEGY_EXCHANGE", os.getenv("EXCHANGE", "NSE"))
FNO_EXCHANGE = os.getenv("FNO_EXCHANGE", "NFO")
INDEX_EXCHANGE = os.getenv("INDEX_EXCHANGE", "NSE_INDEX")
UNDERLYINGS = [
    "NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK",
    "SBIN", "INFY", "HINDALCO", "TCS", "TATAMOTORS", "BHARTIARTL",
]
INDEX_UNDERLYINGS = {
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX", "NIFTYNXT50",
}
RISK_PERCENT = 1.0
ACCOUNT_CAPITAL = float(os.getenv("ACCOUNT_CAPITAL", "1000000"))
PREMIUM_STOP_PCT = 45.0
DTE_MIN = 30
DTE_MAX = 60
DELTA_TARGET = (0.30, 0.45)
ASYM_SCORE_THRESHOLD = 0.65
IVR_BLOCK_THRESHOLD = 40.0

client = api(api_key=API_KEY, host=API_HOST)


def _underlying_exchange(symbol: str) -> str:
    return INDEX_EXCHANGE if symbol in INDEX_UNDERLYINGS else SPOT_EXCHANGE


def _extract_payload(response):
    if not response:
        return None
    if isinstance(response, dict) and "data" in response:
        return response.get("data")
    return response


def _extract_quote_field(response, key: str):
    payload = _extract_payload(response)
    if isinstance(payload, dict):
        return payload.get(key)
    if isinstance(response, dict):
        return response.get(key)
    return None


def calculate_iv_rank(current_iv: float | None, iv_52w_low: float | None, iv_52w_high: float | None) -> float | None:
    if current_iv is None or iv_52w_low is None or iv_52w_high is None:
        return None
    if iv_52w_high <= iv_52w_low:
        return None
    return (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100


def _classify_ce_flow(chain_rows: list[dict]) -> tuple[int, str]:
    ce_oi_chg = sum(r.get("ce_oi_chg", 0) or 0 for r in chain_rows)
    ce_ltp_chg = sum(r.get("ce_ltp_chg", 0) or 0 for r in chain_rows)
    if ce_oi_chg > 0 and ce_ltp_chg > 0.5:
        return 2, "Call Buying"
    if ce_oi_chg < 0 and ce_ltp_chg > 0.5:
        return 1, "CE Short Covering"
    if ce_oi_chg > 0 and ce_ltp_chg < -0.5:
        return -2, "Call Writing"
    if ce_oi_chg < 0 and ce_ltp_chg < -0.5:
        return -1, "CE Long Unwinding"
    return 0, "CE Neutral"


def _classify_pe_flow(chain_rows: list[dict]) -> tuple[int, str]:
    pe_oi_chg = sum(r.get("pe_oi_chg", 0) or 0 for r in chain_rows)
    pe_ltp_chg = sum(r.get("pe_ltp_chg", 0) or 0 for r in chain_rows)
    if pe_oi_chg > 0 and pe_ltp_chg < -0.5:
        return 2, "Put Writing"
    if pe_oi_chg > 0 and pe_ltp_chg > 0.5:
        return -2, "Put Buying"
    if pe_oi_chg < 0 and pe_ltp_chg > 0.5:
        return 1, "PE Short Covering"
    if pe_oi_chg < 0 and pe_ltp_chg < -0.5:
        return -1, "PE Long Unwinding"
    return 0, "PE Neutral"


def _flatten_option_chain(chain_response) -> list[dict]:
    payload = chain_response if isinstance(chain_response, dict) else {}
    chain_rows = payload.get("chain", payload.get("data", []))
    if not isinstance(chain_rows, list):
        return []

    flat_rows: list[dict] = []
    for row in chain_rows:
        strike = row.get("strike")
        if strike is None:
            continue
        ce = row.get("ce") or {}
        pe = row.get("pe") or {}
        flat_rows.append(
            {
                "strike": strike,
                "ce_symbol": ce.get("symbol"),
                "ce_ltp": float(ce.get("ltp") or 0),
                "ce_oi": float(ce.get("oi") or 0),
                "ce_oi_chg": float(ce.get("oi_change") or 0),
                "ce_ltp_chg": float(ce.get("change") or ce.get("ltp_change") or 0),
                "ce_volume": float(ce.get("volume") or 0),
                "ce_delta": abs(float(ce.get("delta") or 0)),
                "ce_lotsize": int(ce.get("lotsize") or 0),
                "pe_symbol": pe.get("symbol"),
                "pe_ltp": float(pe.get("ltp") or 0),
                "pe_oi": float(pe.get("oi") or 0),
                "pe_oi_chg": float(pe.get("oi_change") or 0),
                "pe_ltp_chg": float(pe.get("change") or pe.get("ltp_change") or 0),
                "pe_volume": float(pe.get("volume") or 0),
                "pe_delta": abs(float(pe.get("delta") or 0)),
                "pe_lotsize": int(pe.get("lotsize") or 0),
            }
        )
    return flat_rows


def _select_candidate(flat_rows: list[dict], spot: float, direction: str) -> dict | None:
    candidates: list[dict] = []
    for row in flat_rows:
        strike = float(row["strike"])
        if direction == "bullish":
            premium = row.get("ce_ltp") or 0
            delta = row.get("ce_delta") or 0
            volume = row.get("ce_volume") or 0
            oi = row.get("ce_oi") or 0
            lotsize = row.get("ce_lotsize") or 0
            symbol = row.get("ce_symbol")
            strike_ok = spot <= strike <= spot * 1.05
            option_type = "CE"
        else:
            premium = row.get("pe_ltp") or 0
            delta = row.get("pe_delta") or 0
            volume = row.get("pe_volume") or 0
            oi = row.get("pe_oi") or 0
            lotsize = row.get("pe_lotsize") or 0
            symbol = row.get("pe_symbol")
            strike_ok = spot * 0.95 <= strike <= spot
            option_type = "PE"

        if not strike_ok or not symbol or premium <= 0 or lotsize <= 0:
            continue
        if oi <= 50000 or volume <= 10000:
            continue
        if not (DELTA_TARGET[0] <= delta <= DELTA_TARGET[1]):
            continue

        candidates.append(
            {
                "symbol": symbol,
                "strike": strike,
                "premium": premium,
                "delta": delta,
                "oi": oi,
                "volume": volume,
                "lotsize": lotsize,
                "option_type": option_type,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda row: (-row["oi"], -row["volume"], abs(row["delta"] - 0.35)))
    return candidates[0]


def _size_quantity(best: dict) -> int:
    risk_rupees = ACCOUNT_CAPITAL * RISK_PERCENT / 100
    premium = float(best["premium"])
    lotsize = int(best["lotsize"])
    raw_qty = risk_rupees / premium
    return max(lotsize, int(raw_qty // lotsize) * lotsize)


def check_all_checkpoints(underlying: str):
    """Return ``(passed, signal_dict)`` when the selection filters pass."""
    quote = client.quotes(symbol=underlying, exchange=_underlying_exchange(underlying))
    spot = float(_extract_quote_field(quote, "ltp") or 0)
    if spot <= 0:
        return False, None

    atm_iv = _extract_quote_field(quote, "iv")
    iv_rank = calculate_iv_rank(float(atm_iv) if atm_iv is not None else None, None, None)
    if iv_rank is not None and iv_rank >= IVR_BLOCK_THRESHOLD:
        return False, None

    expiry_response = client.expiry(
        symbol=underlying,
        exchange=FNO_EXCHANGE,
        instrumenttype="options",
    )
    expiry_list = _extract_payload(expiry_response) or []
    if not isinstance(expiry_list, list):
        return False, None

    now = datetime.now().date()
    target_expiry = None
    for exp in expiry_list:
        try:
            dte = (datetime.strptime(str(exp), "%d%b%y").date() - now).days
        except ValueError:
            continue
        if DTE_MIN <= dte <= DTE_MAX:
            target_expiry = str(exp)
            break
    if not target_expiry:
        return False, None

    chain_response = client.optionchain(
        underlying=underlying,
        exchange=_underlying_exchange(underlying),
        expiry_date=target_expiry,
        strike_count=50,
    )
    flat_rows = _flatten_option_chain(chain_response)
    if not flat_rows:
        return False, None

    ce_score, ce_label = _classify_ce_flow(flat_rows)
    pe_score, pe_label = _classify_pe_flow(flat_rows)
    flow_score = ce_score + pe_score
    if flow_score > 0:
        direction = "bullish"
    elif flow_score < 0:
        direction = "bearish"
    else:
        return False, None

    iv_component = 0.175 if iv_rank is None else (1 - iv_rank / 100) * 0.35
    asym_score = iv_component + 0.25 + 0.20 + 0.10 + 0.10
    if asym_score < ASYM_SCORE_THRESHOLD:
        return False, None

    best = _select_candidate(flat_rows, spot, direction)
    if not best:
        return False, None

    qty = _size_quantity(best)
    return True, {
        "underlying": underlying,
        "spot": spot,
        "expiry_date": target_expiry,
        "direction": direction,
        "option_type": best["option_type"],
        "option_symbol": best["symbol"],
        "strike": best["strike"],
        "premium": best["premium"],
        "lotsize": best["lotsize"],
        "quantity": qty,
        "iv_rank": iv_rank,
        "asym_score": round(asym_score, 3),
        "ce_flow": ce_label,
        "pe_flow": pe_label,
    }


def run_strategy():
    print(f"[{datetime.now()}] Checkpoint prototype running (selection only)")
    selected_rows = []
    for underlying in UNDERLYINGS:
        passed, signal = check_all_checkpoints(underlying)
        if passed and signal:
            selected_rows.append(signal)
            print(f"SELECTED → {signal}")

    if not selected_rows:
        print("No candidates passed all checkpoints.")
        return

    print("\nSummary")
    print(pd.DataFrame(selected_rows))
    print("\nExecution is intentionally disabled in this prototype.")


if __name__ == "__main__":
    run_strategy()
