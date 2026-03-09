# app.py
import hmac
from datetime import date
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Multi-Portfolio Tracker (Cash + Snapshot)", layout="wide")

# =========================
# 0) Optional branding
# =========================
LOGO_PATH = Path(__file__).parent / "biglogo-white.png"
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)

# =========================
# 1) Single-login gate
# =========================
def check_password() -> bool:
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if st.session_state.is_admin:
        return True

    with st.sidebar:
        st.markdown("### Admin login")
        pw = st.text_input("Password", type="password")
        if st.button("Log in"):
            secret = st.secrets.get("ADMIN_PASSWORD", "")
            if secret and hmac.compare_digest(pw, secret):
                st.session_state.is_admin = True
                st.success("Logged in.")
            else:
                st.error("Wrong password.")
    return st.session_state.is_admin


is_admin = check_password()

with st.sidebar:
    st.markdown("---")
    st.write("Mode:", "✅ Admin (edit enabled)" if is_admin else "👀 Public (read-only)")
    if is_admin and st.button("Log out"):
        st.session_state.is_admin = False
        st.rerun()

# =========================
# 2) Storage (CSV)
# =========================
TXN_PATH = "transactions.csv"
PORTFOLIO_PATH = "portfolios.csv"
BASELINE_PATH = "baseline_lots.csv"

TXN_COLS = ["txn_id", "portfolio", "date", "type", "ticker", "shares", "price", "amount"]
PORTFOLIO_COLS = ["portfolio", "start_mode", "as_of_date", "starting_cash"]
VALID_MODES = ["ledger_complete", "snapshot_start"]
# NOTE: shares_open can be POSITIVE (long) or NEGATIVE (short) for snapshot portfolios
BASELINE_COLS = ["lot_id", "portfolio", "ticker", "buy_date", "buy_price", "shares_open"]

# Transaction types:
# - buy: open/increase long
# - sell: reduce/close long
# - short: open/increase short (borrow+sell)
# - cover: buy-to-cover reduce/close short
# - dividend: cash dividend
VALID_TXN_TYPES = ["buy", "sell", "short", "cover", "dividend"]
PORTFOLIO_MIN_DATE = date(2000, 1, 1)


def _clean_str(x) -> str:
    return str(x).strip() if x is not None else ""


def load_portfolios() -> pd.DataFrame:
    try:
        df = pd.read_csv(PORTFOLIO_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=PORTFOLIO_COLS)

    for c in PORTFOLIO_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = df[PORTFOLIO_COLS].copy()
    df["portfolio"] = df["portfolio"].astype(str).str.strip()
    df = df[df["portfolio"].notna() & (df["portfolio"] != "")]

    df["start_mode"] = df["start_mode"].astype(str).str.strip().str.lower()
    df.loc[~df["start_mode"].isin(VALID_MODES), "start_mode"] = "ledger_complete"

    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    df["starting_cash"] = pd.to_numeric(df["starting_cash"], errors="coerce").fillna(0.0)

    if "Main" not in set(df["portfolio"].tolist()):
        df = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "portfolio": "Main",
                            "start_mode": "ledger_complete",
                            "as_of_date": pd.to_datetime(date.today()),
                            "starting_cash": 0.0,
                        }
                    ]
                ),
                df,
            ],
            ignore_index=True,
        )

    df.loc[df["as_of_date"].isna(), "as_of_date"] = pd.to_datetime(date.today())
    df = df.drop_duplicates(subset=["portfolio"]).sort_values("portfolio").reset_index(drop=True)
    return df


def save_portfolios(df: pd.DataFrame) -> None:
    out = df.copy()
    out["portfolio"] = out["portfolio"].astype(str).str.strip()
    out["start_mode"] = out["start_mode"].astype(str).str.strip().str.lower()
    out.loc[~out["start_mode"].isin(VALID_MODES), "start_mode"] = "ledger_complete"
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    out.loc[out["as_of_date"].isna(), "as_of_date"] = pd.to_datetime(date.today())
    out["starting_cash"] = pd.to_numeric(out["starting_cash"], errors="coerce").fillna(0.0)
    out = out[out["portfolio"].notna() & (out["portfolio"] != "")]
    out = out.drop_duplicates(subset=["portfolio"]).sort_values("portfolio")
    out.to_csv(PORTFOLIO_PATH, index=False)


def load_txns() -> pd.DataFrame:
    try:
        df = pd.read_csv(TXN_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=TXN_COLS)

    for c in TXN_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = df[TXN_COLS].copy()
    if df.empty:
        return df

    df["portfolio"] = df["portfolio"].astype(str).str.strip()
    df.loc[df["portfolio"].isna() | (df["portfolio"] == ""), "portfolio"] = "Main"

    df["type"] = df["type"].astype(str).str.lower().str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["txn_id"] = df["txn_id"].astype(str)

    df = df.dropna(subset=["portfolio", "date", "type", "txn_id"])
    df = df[df["type"].isin(VALID_TXN_TYPES)].copy()

    is_trade = df["type"].isin(["buy", "sell", "short", "cover"])
    trades = df[is_trade].dropna(subset=["ticker", "shares", "price"]).copy()
    trades = trades[(trades["shares"] > 0) & (trades["price"] >= 0)]

    divs = df[~is_trade].dropna(subset=["amount"]).copy()
    divs = divs[divs["amount"] >= 0]
    divs["ticker"] = divs["ticker"].fillna("").astype(str)

    out = pd.concat([trades, divs], ignore_index=True)
    return out.sort_values(["portfolio", "date", "txn_id"]).reset_index(drop=True)


def save_txns(df: pd.DataFrame) -> None:
    out = df.copy()
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out.to_csv(TXN_PATH, index=False)


def load_baseline() -> pd.DataFrame:
    try:
        df = pd.read_csv(BASELINE_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=BASELINE_COLS)

    for c in BASELINE_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = df[BASELINE_COLS].copy()
    if df.empty:
        return df

    df["lot_id"] = df["lot_id"].astype(str)
    df["portfolio"] = df["portfolio"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce")
    df["shares_open"] = pd.to_numeric(df["shares_open"], errors="coerce")

    df = df.dropna(subset=["lot_id", "portfolio", "ticker", "buy_date", "buy_price", "shares_open"])
    # allow negative shares_open (short baseline), but not zero
    df = df[(df["shares_open"].abs() > 1e-12) & (df["buy_price"] >= 0)]
    return df.sort_values(["portfolio", "ticker", "buy_date", "lot_id"]).reset_index(drop=True)


def save_baseline(df: pd.DataFrame) -> None:
    out = df.copy()
    if not out.empty:
        out["buy_date"] = pd.to_datetime(out["buy_date"]).dt.date.astype(str)
    out.to_csv(BASELINE_PATH, index=False)


# =========================
# 2c) Market data helpers
# =========================
@st.cache_data(ttl=600)
def fetch_last_prices(tickers: list[str]) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)

    data = yf.download(tickers, period="5d", interval="1d", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[-1]
    else:
        close = pd.Series({tickers[0]: data["Close"].iloc[-1]})

    close.index = [str(x).upper() for x in close.index]
    return close.astype(float)


@st.cache_data(ttl=900)
def fetch_price_history(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        start=pd.to_datetime(start).date(),
        end=(pd.to_datetime(end) + pd.Timedelta(days=1)).date(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"].copy()
    else:
        px = pd.DataFrame({tickers[0]: data["Close"]})

    px.index = pd.to_datetime(px.index).normalize()
    px.columns = [str(c).upper() for c in px.columns]
    px = px.sort_index().ffill()
    return px


def fetch_prices_on_or_before(tickers: list[str], d: date) -> pd.Series:
    """
    For each ticker, returns close price on or BEFORE date d.
    If d is today/future, falls back to latest available close.
    """
    cleaned = sorted(set([str(x).upper().strip() for x in tickers if str(x).strip()]))
    if not cleaned:
        return pd.Series(dtype=float)

    target = pd.to_datetime(d).normalize()
    today_n = pd.to_datetime(date.today()).normalize()

    if target >= today_n:
        return fetch_last_prices(cleaned)

    start = target - pd.Timedelta(days=14)
    px = fetch_price_history(cleaned, start=start, end=target)
    if px is None or px.empty:
        return pd.Series(dtype=float)

    px = px[px.index <= target]
    if px.empty:
        return pd.Series(dtype=float)

    return px.iloc[-1].astype(float)


@st.cache_data(ttl=1800)
def fetch_close_on_or_before(ticker: str, d: date) -> float | None:
    """
    Close price on or BEFORE date d (handles weekends/holidays).
    """
    t = str(ticker).upper().strip()
    if not t:
        return None

    target = pd.to_datetime(d).normalize()
    start = (target - pd.Timedelta(days=10)).date()
    end = (target + pd.Timedelta(days=1)).date()

    try:
        df = yf.download([t], start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        close = df["Close"] if not isinstance(df.columns, pd.MultiIndex) else df["Close"]
        if isinstance(close, pd.DataFrame):
            s = close[t]
        else:
            s = close
        s.index = pd.to_datetime(s.index).normalize()
        s = s.dropna().sort_index()
        s = s[s.index <= target]
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_dividends_series(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    yfinance dividends sometimes come back tz-aware; we normalize to tz-naive midnight.
    Returns dividend per share events between [start, end].
    """
    t = str(ticker).upper().strip()
    s = pd.to_datetime(start).normalize()
    e = pd.to_datetime(end).normalize()

    div = yf.Ticker(t).dividends
    if div is None or len(div) == 0:
        return pd.Series(dtype=float, name=t)

    idx = pd.to_datetime(div.index)
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass

    idx = pd.to_datetime(idx).normalize()
    out = pd.Series(div.values, index=idx, name=t).sort_index()
    out = out[(out.index >= s) & (out.index <= e)].copy()
    return out.astype(float)


# =========================
# 2b) Yahoo sector/industry (cached)
# =========================
@st.cache_data(ttl=86400)
def fetch_sector_industry(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in sorted(set([str(x).upper().strip() for x in tickers if str(x).strip()])):
        try:
            info = yf.Ticker(t).info or {}
            sector = info.get("sector") or "Unknown"
            industry = info.get("industry") or "Unknown"
        except Exception:
            sector, industry = "Unknown", "Unknown"
        rows.append({"ticker": t, "sector": sector, "industry": industry})
    return pd.DataFrame(rows)


# =========================
# 2d) Bulk upload helpers (CSV)
# =========================
def _parse_number_allow_parens(x):
    """
    Accepts numbers like:
      10, -10, (10), "  (10)  "
    Returns float or np.nan.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace(",", "")
    try:
        v = float(s)
        if neg:
            v = -v
        return v
    except Exception:
        return np.nan


def parse_simple_txn_csv(uploaded_file) -> pd.DataFrame:
    """
    Required columns:
      TICKER | TRANSACTION TYPE | DATE | SHARE COUNT

    Optional columns:
      PRICE   (if omitted/blank => filled from Yahoo close-on-or-before date)

    Returns normalized df:
      ticker, type, date, shares, price
    Only keeps type in {buy, sell, short, cover}.
    """
    df = pd.read_csv(uploaded_file)
    df.columns = [str(c).strip().upper() for c in df.columns]

    required = ["TICKER", "TRANSACTION TYPE", "DATE", "SHARE COUNT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame()
    out["ticker"] = df["TICKER"].astype(str).str.upper().str.strip()
    out["type"] = df["TRANSACTION TYPE"].astype(str).str.lower().str.strip()
    out["date"] = pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()

    shares_raw = df["SHARE COUNT"].apply(_parse_number_allow_parens)
    out["shares"] = pd.to_numeric(shares_raw, errors="coerce").abs()

    if "PRICE" in df.columns:
        out["price"] = pd.to_numeric(df["PRICE"].apply(_parse_number_allow_parens), errors="coerce")
        out.loc[out["price"] < 0, "price"] = np.nan
    else:
        out["price"] = np.nan

    out = out[out["type"].isin(["buy", "sell", "short", "cover"])].copy()
    out = out.dropna(subset=["ticker", "type", "date", "shares"])
    out = out[(out["ticker"] != "") & (out["shares"] > 0)]
    return out.reset_index(drop=True)


def enrich_import_with_prices(import_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensures there is a valid 'price' for every row.
    - If import_df has a non-null price, we keep it.
    - Otherwise we fetch close-on-or-before (ticker, date).

    Returns: (good_df, bad_df)
    """
    if import_df is None or import_df.empty:
        return import_df, pd.DataFrame()

    rows = []
    failed = []

    for _, r in import_df.iterrows():
        t = str(r.get("ticker", "")).upper().strip()
        d = pd.to_datetime(r.get("date")).date() if pd.notna(r.get("date")) else None
        sh = float(r.get("shares", np.nan)) if pd.notna(r.get("shares")) else np.nan
        typ = str(r.get("type", "")).lower().strip()

        if not t or d is None or not np.isfinite(sh) or sh <= 0 or typ not in ["buy", "sell", "short", "cover"]:
            failed.append({**r.to_dict(), "reason": "Invalid ticker/date/shares/type"})
            continue

        px_in = r.get("price", np.nan)
        if pd.notna(px_in) and np.isfinite(float(px_in)) and float(px_in) >= 0:
            rows.append({**r.to_dict(), "price": float(px_in)})
            continue

        px = fetch_close_on_or_before(t, d)
        if px is None or (not np.isfinite(px)):
            failed.append({**r.to_dict(), "reason": "Could not fetch close on/before date"})
            continue

        rows.append({**r.to_dict(), "price": float(px)})

    good = pd.DataFrame(rows)
    bad = pd.DataFrame(failed)
    return good, bad


# =========================
# 3) Lot engine + accounting (LONG + SHORT)
# =========================
def apply_reduce_to_lots(lots: list[dict], reduce_shares: float, method: str, side: str):
    """
    Reduce existing exposure by matching lots.

    side="LONG": reduce positive shares by consuming positive lots toward 0
    side="SHORT": reduce short by consuming negative lots toward 0 (increase toward 0)
    """
    realized_rows = []
    remaining = float(reduce_shares)

    if method.upper() == "FIFO":
        lot_iter = lots
    else:
        lot_iter = list(reversed(lots))

    i = 0
    while remaining > 1e-12 and i < len(lot_iter):
        lot = lot_iter[i]
        sh_open = float(lot["shares_open"])

        if side == "LONG":
            if sh_open <= 1e-12:
                i += 1
                continue
            take = min(remaining, sh_open)
            lot["shares_open"] = sh_open - take
            remaining -= take
            realized_rows.append((lot, take))

        else:  # SHORT
            if sh_open >= -1e-12:
                i += 1
                continue
            avail = abs(sh_open)
            take = min(remaining, avail)
            lot["shares_open"] = sh_open + take
            remaining -= take
            realized_rows.append((lot, take))

        i += 1

    return realized_rows, remaining


def build_lots_with_baseline(trades: pd.DataFrame, baseline: pd.DataFrame, method: str):
    """
    Lots support:
      - shares_open > 0 : long lots (entry = buy_price)
      - shares_open < 0 : short lots (entry = short_price stored in buy_price)
    Realized PnL:
      - LONG: shares_sold * (sell_price - entry)
      - SHORT: shares_covered * (entry - cover_price)
    """
    open_cols = ["lot_id", "ticker", "buy_date", "buy_price", "shares_open"]
    real_cols = [
        "sale_id",
        "ticker",
        "position",
        "buy_date",
        "buy_price",
        "sell_date",
        "sell_price",
        "shares_sold",
        "pnl",
    ]

    lots_by_ticker: dict[str, list[dict]] = {}

    if not baseline.empty:
        for _, r in baseline.iterrows():
            t = str(r["ticker"]).upper().strip()
            lots_by_ticker.setdefault(t, []).append(
                {
                    "lot_id": str(r["lot_id"]),
                    "ticker": t,
                    "buy_date": pd.to_datetime(r["buy_date"]).date(),
                    "buy_price": float(r["buy_price"]),
                    "shares_open": float(r["shares_open"]),
                }
            )

    realized = []

    if trades.empty:
        open_lots = []
        for _, lots in lots_by_ticker.items():
            open_lots.extend([x for x in lots if abs(float(x["shares_open"])) > 1e-12])
        return pd.DataFrame(open_lots, columns=open_cols), pd.DataFrame(columns=real_cols)

    trades = trades.copy().sort_values(["ticker", "date", "txn_id"])
    for _, r in trades.iterrows():
        typ = str(r["type"]).lower()
        tkr = str(r["ticker"]).upper().strip()
        sh = float(r["shares"])
        px = float(r["price"])
        dt = pd.to_datetime(r["date"]).date()
        tid = str(r["txn_id"])

        lots_by_ticker.setdefault(tkr, [])

        if typ == "buy":
            lots_by_ticker[tkr].append(
                {"lot_id": tid, "ticker": tkr, "buy_date": dt, "buy_price": px, "shares_open": +sh}
            )

        elif typ == "sell":
            matches, _ = apply_reduce_to_lots(lots_by_ticker[tkr], sh, method, side="LONG")
            for lot, shares_sold in matches:
                pnl = shares_sold * (px - float(lot["buy_price"]))
                realized.append(
                    {
                        "sale_id": tid,
                        "ticker": tkr,
                        "position": "LONG",
                        "buy_date": lot["buy_date"],
                        "buy_price": float(lot["buy_price"]),
                        "sell_date": dt,
                        "sell_price": px,
                        "shares_sold": shares_sold,
                        "pnl": pnl,
                    }
                )

        elif typ == "short":
            lots_by_ticker[tkr].append(
                {"lot_id": tid, "ticker": tkr, "buy_date": dt, "buy_price": px, "shares_open": -sh}
            )

        elif typ == "cover":
            matches, _ = apply_reduce_to_lots(lots_by_ticker[tkr], sh, method, side="SHORT")
            for lot, shares_cov in matches:
                entry = float(lot["buy_price"])
                pnl = shares_cov * (entry - px)
                realized.append(
                    {
                        "sale_id": tid,
                        "ticker": tkr,
                        "position": "SHORT",
                        "buy_date": lot["buy_date"],
                        "buy_price": entry,
                        "sell_date": dt,
                        "sell_price": px,
                        "shares_sold": shares_cov,
                        "pnl": pnl,
                    }
                )

    open_lots = []
    for _, lots in lots_by_ticker.items():
        open_lots.extend([x for x in lots if abs(float(x["shares_open"])) > 1e-12])

    return pd.DataFrame(open_lots, columns=open_cols), pd.DataFrame(realized, columns=real_cols)


def cash_delta(row: pd.Series) -> float:
    typ = str(row["type"]).lower()
    if typ == "buy":
        return -float(row["shares"]) * float(row["price"])
    if typ == "sell":
        return +float(row["shares"]) * float(row["price"])
    if typ == "short":
        return +float(row["shares"]) * float(row["price"])
    if typ == "cover":
        return -float(row["shares"]) * float(row["price"])
    if typ == "dividend":
        return +float(row["amount"])
    return 0.0


def get_portfolio_meta(portfolios_df: pd.DataFrame, portfolio: str):
    r = portfolios_df.loc[portfolios_df["portfolio"] == portfolio].iloc[0]
    return {
        "start_mode": str(r["start_mode"]),
        "as_of_date": pd.to_datetime(r["as_of_date"]),
        "starting_cash": float(r["starting_cash"]),
    }


def validate_candidate_state(
    portfolio_meta: dict,
    portfolio_txns_all: pd.DataFrame,
    portfolio_baseline: pd.DataFrame,
    method: str,
) -> tuple[bool, str]:
    """
    Rules:
    - Snapshot portfolios: no txns before as_of_date
    - No negative CASH (still enforced: no margin loans)
    - SELL can only reduce existing LONG shares
    - COVER can only reduce existing SHORT shares
    - SHORT can open/increase shorts (no cap here; cash constraint still applies)
    """
    start_mode = portfolio_meta["start_mode"]
    as_of = pd.to_datetime(portfolio_meta["as_of_date"])
    starting_cash = float(portfolio_meta["starting_cash"])

    txns = portfolio_txns_all.copy()
    if not txns.empty:
        txns["date"] = pd.to_datetime(txns["date"])
        txns = txns.sort_values(["date", "txn_id"])

    if start_mode == "snapshot_start" and not txns.empty:
        if (txns["date"] < as_of).any():
            return False, f"Snapshot portfolios cannot have transactions before as-of date ({as_of.date()})."

    cash = starting_cash
    for _, r in txns.iterrows():
        cash += cash_delta(r)
        if cash < -1e-9:
            return False, "Invalid: cash would go negative (margin not allowed)."

    trades = txns[txns["type"].isin(["buy", "sell", "short", "cover"])].copy()
    if trades.empty:
        return True, ""

    long_shares = {}
    short_shares = {}

    if not portfolio_baseline.empty:
        for _, r in portfolio_baseline.iterrows():
            t = str(r["ticker"]).upper().strip()
            sh0 = float(r["shares_open"])
            if sh0 > 0:
                long_shares[t] = long_shares.get(t, 0.0) + sh0
            elif sh0 < 0:
                short_shares[t] = short_shares.get(t, 0.0) + abs(sh0)

    for _, r in trades.sort_values(["date", "txn_id"]).iterrows():
        t = str(r["ticker"]).upper().strip()
        sh = float(r["shares"])
        typ = str(r["type"]).lower()

        if typ == "buy":
            long_shares[t] = long_shares.get(t, 0.0) + sh

        elif typ == "sell":
            if long_shares.get(t, 0.0) + 1e-12 < sh:
                return False, f"Invalid SELL: not enough LONG shares of {t} to sell {sh:.3f}."
            long_shares[t] -= sh

        elif typ == "short":
            short_shares[t] = short_shares.get(t, 0.0) + sh

        elif typ == "cover":
            if short_shares.get(t, 0.0) + 1e-12 < sh:
                return False, f"Invalid COVER: not enough SHORT shares of {t} to cover {sh:.3f}."
            short_shares[t] -= sh

    return True, ""


def portfolio_snapshot(
    portfolio_meta: dict,
    txns: pd.DataFrame,
    baseline: pd.DataFrame,
    method: str,
    valuation_date: date | None = None,
):
    as_of = pd.to_datetime(portfolio_meta["as_of_date"])
    start_mode = portfolio_meta["start_mode"]
    starting_cash = float(portfolio_meta["starting_cash"])

    eval_date = pd.to_datetime(valuation_date if valuation_date is not None else date.today()).normalize()

    df = txns.copy()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= eval_date].copy()

    if start_mode == "snapshot_start" and not df.empty:
        df = df[df["date"] >= as_of].copy()

    cash_now = starting_cash + (df.apply(cash_delta, axis=1).sum() if not df.empty else 0.0)

    trades = (
        df[df["type"].isin(["buy", "sell", "short", "cover"])].copy()
        if not df.empty
        else pd.DataFrame(columns=TXN_COLS)
    )
    open_lots, realized = build_lots_with_baseline(trades, baseline, method)

    if not open_lots.empty:
        tickers = sorted(open_lots["ticker"].unique().tolist())
        live = fetch_prices_on_or_before(tickers, eval_date.date())

        lots_view = open_lots.copy()
        lots_view["last_price"] = lots_view["ticker"].map(live).astype(float)

        lots_view["market_value"] = lots_view["shares_open"] * lots_view["last_price"]
        lots_view["unrealized_pnl"] = lots_view["shares_open"] * (lots_view["last_price"] - lots_view["buy_price"])

        lots_view["position"] = np.where(lots_view["shares_open"] >= 0, "LONG", "SHORT")
        lots_view["unrealized_return_%"] = np.where(
            lots_view["buy_price"] > 0,
            ((lots_view["last_price"] / lots_view["buy_price"]) - 1.0) * 100.0,
            np.nan,
        )

        tmp = lots_view.copy()
        tmp["abs_shares"] = tmp["shares_open"].abs()
        tmp["cost_dollars"] = tmp["abs_shares"] * tmp["buy_price"]

        holdings = tmp.groupby("ticker", as_index=False).agg(
            shares=("shares_open", "sum"),
            abs_shares=("abs_shares", "sum"),
            cost_dollars=("cost_dollars", "sum"),
            market_value=("market_value", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        holdings["position"] = np.where(holdings["shares"] >= 0, "LONG", "SHORT")
        holdings["avg_cost"] = np.where(
            holdings["abs_shares"] > 0, holdings["cost_dollars"] / holdings["abs_shares"], np.nan
        )
        holdings = holdings.sort_values("market_value", ascending=False)
    else:
        lots_view = open_lots
        holdings = pd.DataFrame()

    mv = float(lots_view["market_value"].sum()) if not lots_view.empty else 0.0
    nav = float(cash_now + mv)
    unreal = float(lots_view["unrealized_pnl"].sum()) if not lots_view.empty else 0.0
    real = float(realized["pnl"].sum()) if not realized.empty else 0.0

    return {
        "cash": float(cash_now),
        "market_value": mv,
        "nav": nav,
        "unrealized_pnl": unreal,
        "realized_pnl": real,
        "lots": lots_view,
        "realized": realized,
        "holdings": holdings,
        "filtered_txns": df.sort_values("date", ascending=False) if not df.empty else df,
        "as_of": as_of,
        "start_mode": start_mode,
        "starting_cash": starting_cash,
    }


# =========================
# 4) Chart helpers (WEEKLY/Daily)
# =========================
def _portfolio_start_date(meta: dict) -> pd.Timestamp:
    return pd.to_datetime(meta["as_of_date"]).normalize()


def index_to_100(series: pd.Series) -> pd.Series:
    s = series.copy()
    first = s.first_valid_index()
    if first is None:
        return s
    base = float(s.loc[first])
    if base == 0:
        return s
    return (s / base) * 100.0


def compute_nav_series_for_portfolio(
    pname: str,
    meta: dict,
    txns_all: pd.DataFrame,
    baseline_all: pd.DataFrame,
    end_date: pd.Timestamp,
    chart_freq: str,  # "D" or "W-MON"
) -> pd.Series:
    start = _portfolio_start_date(meta)
    end = pd.to_datetime(end_date).normalize()

    txns = txns_all[txns_all["portfolio"] == pname].copy()
    if not txns.empty:
        txns["date"] = pd.to_datetime(txns["date"]).dt.normalize()
        txns = txns.sort_values(["date", "txn_id"])
        txns = txns[txns["date"] >= start].copy()

    baseline = baseline_all[baseline_all["portfolio"] == pname].copy()
    if not baseline.empty:
        baseline["ticker"] = baseline["ticker"].astype(str).str.upper().str.strip()

    if chart_freq == "D":
        idx = pd.date_range(start=start, end=end, freq="D")
    else:
        idx_week = pd.date_range(start=start, end=end, freq=chart_freq)
        idx = pd.DatetimeIndex(sorted(set([start] + list(idx_week))))

    if len(idx) == 0:
        idx = pd.DatetimeIndex([start])

    starting_cash = float(meta["starting_cash"])
    if txns.empty:
        cash = pd.Series(starting_cash, index=idx)
    else:
        cf = txns.apply(cash_delta, axis=1)
        cf_by_day = cf.groupby(txns["date"]).sum()

        if chart_freq == "D":
            cash_flow = cf_by_day.reindex(idx, fill_value=0.0)
        else:
            cf_week = cf_by_day.groupby(pd.Grouper(freq="W-MON")).sum()
            cash_flow = cf_week.reindex(idx, fill_value=0.0)

        cash = starting_cash + cash_flow.cumsum()

    baseline_shares = {}
    if not baseline.empty:
        for _, r in baseline.iterrows():
            t = str(r["ticker"]).upper().strip()
            baseline_shares[t] = baseline_shares.get(t, 0.0) + float(r["shares_open"])

    trades = (
        txns[txns["type"].isin(["buy", "sell", "short", "cover"])].copy()
        if not txns.empty
        else pd.DataFrame(columns=TXN_COLS)
    )
    tickers = set(baseline_shares.keys())

    if not trades.empty:
        trades["ticker"] = trades["ticker"].astype(str).str.upper().str.strip()
        trades["signed_shares"] = np.select(
            [
                trades["type"] == "buy",
                trades["type"] == "sell",
                trades["type"] == "short",
                trades["type"] == "cover",
            ],
            [
                +trades["shares"],
                -trades["shares"],
                -trades["shares"],
                +trades["shares"],
            ],
            default=0.0,
        )
        tickers.update([t for t in trades["ticker"].unique() if t])

    tickers = sorted([t for t in tickers if t])
    if not tickers:
        nav = cash.copy()
        nav.name = pname
        return nav

    shares_df = pd.DataFrame(0.0, index=idx, columns=tickers)

    for t, sh in baseline_shares.items():
        if t in shares_df.columns:
            shares_df.loc[idx[0], t] += sh

    if not trades.empty:
        for t in tickers:
            t_tr = trades[trades["ticker"] == t]
            if t_tr.empty:
                continue

            delta_by_day = t_tr.groupby("date")["signed_shares"].sum()

            if chart_freq == "D":
                shares_df[t] += delta_by_day.reindex(idx, fill_value=0.0)
            else:
                delta_week = delta_by_day.groupby(pd.Grouper(freq="W-MON")).sum()
                shares_df[t] += delta_week.reindex(idx, fill_value=0.0)

    shares_df = shares_df.cumsum()

    px_daily = fetch_price_history(tickers, start, end)
    if px_daily.empty:
        nav = cash.copy()
        nav.name = pname
        return nav

    daily_idx = pd.date_range(start=start, end=end, freq="D")
    px_daily = px_daily.reindex(daily_idx).ffill()
    px = px_daily.reindex(idx).ffill()

    mv = (shares_df * px[tickers]).sum(axis=1)
    nav = cash + mv
    nav.name = pname
    return nav


# =========================
# 4b) Tier-1 analytics helpers
# =========================
def compute_drawdown(nav: pd.Series) -> pd.Series:
    s = nav.dropna().copy()
    if s.empty:
        return pd.Series(dtype=float)
    peak = s.cummax()
    dd = (s / peak) - 1.0
    dd.name = "drawdown"
    return dd


def compute_beta_alpha(port_nav: pd.Series, mkt_px: pd.Series) -> dict:
    rp = port_nav.pct_change().replace([np.inf, -np.inf], np.nan)
    rm = mkt_px.pct_change().replace([np.inf, -np.inf], np.nan)

    df = pd.concat([rp.rename("rp"), rm.rename("rm")], axis=1).dropna()
    if len(df) < 3:
        return {"beta": np.nan, "alpha_week": np.nan, "alpha_ann": np.nan, "r2": np.nan, "n": len(df)}

    cov = float(df["rp"].cov(df["rm"]))
    var = float(df["rm"].var())
    beta = np.nan if var == 0 else cov / var

    alpha_week = float(df["rp"].mean() - beta * df["rm"].mean())
    alpha_ann = float((1.0 + alpha_week) ** 52 - 1.0)

    corr = float(df["rp"].corr(df["rm"]))
    r2 = corr * corr if np.isfinite(corr) else np.nan

    return {"beta": beta, "alpha_week": alpha_week, "alpha_ann": alpha_ann, "r2": r2, "n": len(df)}


def rolling_beta_alpha(port_nav: pd.Series, mkt_px: pd.Series, window: int = 26) -> pd.DataFrame:
    rp = port_nav.pct_change().replace([np.inf, -np.inf], np.nan)
    rm = mkt_px.pct_change().replace([np.inf, -np.inf], np.nan)
    df = pd.concat([rp.rename("rp"), rm.rename("rm")], axis=1).dropna()
    if df.empty or len(df) < window:
        return pd.DataFrame(columns=["beta", "alpha_ann"])

    betas = []
    alphas = []
    idx = []
    for i in range(window, len(df) + 1):
        w = df.iloc[i - window : i]
        cov = float(w["rp"].cov(w["rm"]))
        var = float(w["rm"].var())
        beta = np.nan if var == 0 else cov / var
        alpha_week = float(w["rp"].mean() - beta * w["rm"].mean())
        alpha_ann = float((1.0 + alpha_week) ** 52 - 1.0)
        betas.append(beta)
        alphas.append(alpha_ann)
        idx.append(w.index[-1])

    out = pd.DataFrame({"beta": betas, "alpha_ann": alphas}, index=pd.to_datetime(idx))
    return out


def build_allocation_tables(snap: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For portfolios with shorts, signed MV can be negative, so allocations use ABS exposure.
    """
    rows = []

    if not snap["holdings"].empty:
        for _, r in snap["holdings"].iterrows():
            t = str(r["ticker"]).upper().strip()
            mv = float(r["market_value"])
            rows.append({"ticker": t, "label": t, "market_value": mv, "exposure": abs(mv)})

    cash = float(snap["cash"])
    rows.append({"ticker": "", "label": "CASH", "market_value": cash, "exposure": abs(cash)})

    df = pd.DataFrame(rows)
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)
    df["exposure"] = pd.to_numeric(df["exposure"], errors="coerce").fillna(0.0)

    total_exp = float(df["exposure"].sum())
    df["weight"] = 0.0 if total_exp <= 0 else (df["exposure"] / total_exp)

    tickers = sorted([t for t in df["ticker"].unique().tolist() if t])
    meta = fetch_sector_industry(tickers) if tickers else pd.DataFrame(columns=["ticker", "sector", "industry"])

    df = df.merge(meta, on="ticker", how="left")
    df.loc[df["label"] == "CASH", "sector"] = "Cash"
    df.loc[df["label"] == "CASH", "industry"] = "Cash"
    df["sector"] = df["sector"].fillna("Unknown")
    df["industry"] = df["industry"].fillna("Unknown")

    sector_alloc = (
        df.groupby(["sector"], as_index=False).agg(exposure=("exposure", "sum")).sort_values("exposure", ascending=False)
    )
    sector_total = float(sector_alloc["exposure"].sum())
    sector_alloc["weight"] = np.where(sector_total > 0, sector_alloc["exposure"] / sector_total, 0.0)

    industry_alloc = (
        df.groupby(["sector", "industry"], as_index=False)
        .agg(exposure=("exposure", "sum"))
        .sort_values(["sector", "exposure"], ascending=[True, False])
    )
    ind_total = float(industry_alloc["exposure"].sum())
    industry_alloc["weight"] = np.where(ind_total > 0, industry_alloc["exposure"] / ind_total, 0.0)

    return sector_alloc, industry_alloc


def build_contribution_table(snap: dict) -> pd.DataFrame:
    unreal = pd.DataFrame(columns=["ticker", "unrealized_pnl", "unrealized_cost_basis"])
    if not snap["lots"].empty:
        u = snap["lots"].copy()
        u["unrealized_cost_basis"] = u["buy_price"].abs() * u["shares_open"].abs()
        unreal = u.groupby("ticker", as_index=False).agg(
            unrealized_pnl=("unrealized_pnl", "sum"),
            unrealized_cost_basis=("unrealized_cost_basis", "sum"),
        )

    realized = pd.DataFrame(columns=["ticker", "realized_pnl", "realized_cost_basis"])
    if not snap["realized"].empty:
        r = snap["realized"].copy()
        r["realized_cost_basis"] = r["buy_price"].abs() * r["shares_sold"].abs()
        realized = r.groupby("ticker", as_index=False).agg(
            realized_pnl=("pnl", "sum"),
            realized_cost_basis=("realized_cost_basis", "sum"),
        )

    divs = pd.DataFrame(columns=["ticker", "dividend_pnl"])
    tx = snap["filtered_txns"]
    if tx is not None and not tx.empty:
        d = tx[tx["type"] == "dividend"].copy()
        if not d.empty:
            d["ticker"] = d["ticker"].fillna("").astype(str).str.upper().str.strip()
            d.loc[d["ticker"] == "", "ticker"] = "CASH"
            divs = d.groupby("ticker", as_index=False).agg(dividend_pnl=("amount", "sum"))

    out = unreal.merge(realized, on="ticker", how="outer").merge(divs, on="ticker", how="outer")
    if out.empty:
        return out

    out = out.fillna(0.0)
    out["unrealized_return_pct"] = np.where(
        out["unrealized_cost_basis"] > 0,
        (out["unrealized_pnl"] / out["unrealized_cost_basis"]) * 100.0,
        np.nan,
    )
    out["realized_return_pct"] = np.where(
        out["realized_cost_basis"] > 0,
        (out["realized_pnl"] / out["realized_cost_basis"]) * 100.0,
        np.nan,
    )
    out["total_contribution"] = out["unrealized_pnl"] + out["realized_pnl"] + out["dividend_pnl"]
    out = out[
        [
            "ticker",
            "unrealized_pnl",
            "unrealized_return_pct",
            "realized_pnl",
            "realized_return_pct",
            "dividend_pnl",
            "total_contribution",
        ]
    ]
    out = out.sort_values("total_contribution", ascending=False).reset_index(drop=True)
    return out


def build_price_breakdown_table(snap: dict, valuation_date: date | None = None) -> pd.DataFrame:
def build_price_breakdown_table(snap: dict) -> pd.DataFrame:
    entry_rows = []

    if snap.get("lots") is not None and not snap["lots"].empty:
        l = snap["lots"][["ticker", "buy_price", "shares_open"]].copy()
        l["shares_ref"] = l["shares_open"].abs()
        l = l[l["shares_ref"] > 0]
        if not l.empty:
            entry_rows.append(l[["ticker", "buy_price", "shares_ref"]].rename(columns={"buy_price": "entry_price"}))

    if snap.get("realized") is not None and not snap["realized"].empty:
        r = snap["realized"][["ticker", "buy_price", "shares_sold", "sell_price"]].copy()
        r["shares_ref"] = pd.to_numeric(r["shares_sold"], errors="coerce").fillna(0.0).abs()
        r = r[r["shares_ref"] > 0]
        if not r.empty:
            entry_rows.append(r[["ticker", "buy_price", "shares_ref"]].rename(columns={"buy_price": "entry_price"}))

    if entry_rows:
        entry = pd.concat(entry_rows, ignore_index=True)
        entry["weighted"] = entry["entry_price"] * entry["shares_ref"]
        buy_avg = (
            entry.groupby("ticker", as_index=False)
            .agg(total_shares=("shares_ref", "sum"), weighted=("weighted", "sum"))
        )
        buy_avg["avg_buy_price"] = np.where(buy_avg["total_shares"] > 0, buy_avg["weighted"] / buy_avg["total_shares"], np.nan)
        buy_avg = buy_avg[["ticker", "avg_buy_price"]]
    else:
        buy_avg = pd.DataFrame(columns=["ticker", "avg_buy_price"])

    if snap.get("realized") is not None and not snap["realized"].empty:
        sold = snap["realized"][["ticker", "sell_price", "shares_sold"]].copy()
        sold["shares_ref"] = pd.to_numeric(sold["shares_sold"], errors="coerce").fillna(0.0).abs()
        sold = sold[sold["shares_ref"] > 0]
        if not sold.empty:
            sold["weighted"] = sold["sell_price"] * sold["shares_ref"]
            sold_avg = (
                sold.groupby("ticker", as_index=False)
                .agg(total_shares=("shares_ref", "sum"), weighted=("weighted", "sum"))
            )
            sold_avg["avg_sold_price"] = np.where(
                sold_avg["total_shares"] > 0,
                sold_avg["weighted"] / sold_avg["total_shares"],
                np.nan,
            )
            sold_avg = sold_avg[["ticker", "avg_sold_price"]]
        else:
            sold_avg = pd.DataFrame(columns=["ticker", "avg_sold_price"])
    else:
        sold_avg = pd.DataFrame(columns=["ticker", "avg_sold_price"])

    tickers = set(buy_avg.get("ticker", pd.Series(dtype=str)).tolist()) | set(sold_avg.get("ticker", pd.Series(dtype=str)).tolist())
    tx = snap.get("filtered_txns")
    if tx is not None and not tx.empty:
        tx_tickers = tx[tx["type"].isin(["buy", "sell", "short", "cover"])]["ticker"].astype(str).str.upper().str.strip()
        tickers |= set([t for t in tx_tickers.tolist() if t])

    tickers = sorted(tickers)
    if not tickers:
        return pd.DataFrame(columns=["ticker", "avg_buy_price", "avg_sold_price", "current_price"])

    out = pd.DataFrame({"ticker": tickers})
    out = out.merge(buy_avg, on="ticker", how="left").merge(sold_avg, on="ticker", how="left")

    eval_date = valuation_date if valuation_date is not None else date.today()
    live = fetch_prices_on_or_before(tickers, eval_date)
    live = fetch_last_prices(tickers)
    out["current_price"] = out["ticker"].map(live)

    return out.sort_values("ticker").reset_index(drop=True)


def render_sector_pie(sector_alloc: pd.DataFrame, title: str):
    if sector_alloc.empty or sector_alloc["exposure"].sum() <= 0:
        st.info("No sector allocation available yet.")
        return
    fig, ax = plt.subplots()
    ax.pie(sector_alloc["exposure"], labels=sector_alloc["sector"], autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    ax.axis("equal")
    st.pyplot(fig, clear_figure=True)


# =========================
# 4c) Dividend tracker (quarterly accrual)
# =========================
def compute_daily_shares_df(
    pname: str,
    meta: dict,
    txns_all: pd.DataFrame,
    baseline_all: pd.DataFrame,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Daily NET shares outstanding by ticker from start boundary to end_date.
    Shorts are negative shares; dividend accrual uses max(shares, 0) (i.e., ignores borrow costs).
    """
    start = _portfolio_start_date(meta)
    end = pd.to_datetime(end_date).normalize()
    idx = pd.date_range(start=start, end=end, freq="D")
    if len(idx) == 0:
        idx = pd.DatetimeIndex([start])

    txns = txns_all[txns_all["portfolio"] == pname].copy()
    if not txns.empty:
        txns["date"] = pd.to_datetime(txns["date"]).dt.normalize()
        txns = txns.sort_values(["date", "txn_id"])
        txns = txns[txns["date"] >= start].copy()

    baseline = baseline_all[baseline_all["portfolio"] == pname].copy()
    baseline_shares = {}
    if not baseline.empty:
        baseline["ticker"] = baseline["ticker"].astype(str).str.upper().str.strip()
        for _, r in baseline.iterrows():
            t = str(r["ticker"]).upper().strip()
            baseline_shares[t] = baseline_shares.get(t, 0.0) + float(r["shares_open"])

    trades = (
        txns[txns["type"].isin(["buy", "sell", "short", "cover"])].copy()
        if not txns.empty
        else pd.DataFrame(columns=TXN_COLS)
    )
    tickers = set(baseline_shares.keys())
    if not trades.empty:
        trades["ticker"] = trades["ticker"].astype(str).str.upper().str.strip()
        trades["signed_shares"] = np.select(
            [
                trades["type"] == "buy",
                trades["type"] == "sell",
                trades["type"] == "short",
                trades["type"] == "cover",
            ],
            [
                +trades["shares"],
                -trades["shares"],
                -trades["shares"],
                +trades["shares"],
            ],
            default=0.0,
        )
        tickers.update([t for t in trades["ticker"].unique() if t])

    tickers = sorted([t for t in tickers if t])
    if not tickers:
        return pd.DataFrame(index=idx)

    shares_df = pd.DataFrame(0.0, index=idx, columns=tickers)
    for t, sh in baseline_shares.items():
        if t in shares_df.columns:
            shares_df.loc[idx[0], t] += sh

    if not trades.empty:
        for t in tickers:
            t_tr = trades[trades["ticker"] == t]
            if t_tr.empty:
                continue
            delta_by_day = t_tr.groupby("date")["signed_shares"].sum()
            shares_df[t] += delta_by_day.reindex(idx, fill_value=0.0)

    shares_df = shares_df.cumsum()
    return shares_df


def compute_dividend_accrual_quarterly(
    pname: str,
    meta: dict,
    txns_all: pd.DataFrame,
    baseline_all: pd.DataFrame,
    end_date: pd.Timestamp,
) -> dict:
    """
    Uses yfinance dividend-per-share events and multiplies by LONG shares held on that date.
    (Short borrow costs / dividend-in-lieu are not modeled.)
    """
    start = _portfolio_start_date(meta)
    end = pd.to_datetime(end_date).normalize()

    shares_df = compute_daily_shares_df(pname, meta, txns_all, baseline_all, end)
    if shares_df.empty or shares_df.shape[1] == 0:
        return {"total": 0.0, "events": pd.DataFrame(), "quarterly": pd.DataFrame()}

    tickers = list(shares_df.columns)
    events = []

    for t in tickers:
        div = fetch_dividends_series(t, start, end)
        if div is None or div.empty:
            continue

        for d, v in div.items():
            d = pd.to_datetime(d).normalize()
            if d not in shares_df.index:
                continue
            sh_net = float(shares_df.at[d, t])
            sh_long = max(sh_net, 0.0)
            if sh_long <= 1e-12:
                continue
            events.append(
                {
                    "ticker": t,
                    "div_date": d,
                    "div_per_share": float(v),
                    "shares": sh_long,
                    "div_cash": float(v) * sh_long,
                    "quarter": str(d.to_period("Q")),
                    "quarter_end": d.to_period("Q").end_time.normalize(),
                }
            )

    if not events:
        return {"total": 0.0, "events": pd.DataFrame(), "quarterly": pd.DataFrame()}

    ev = pd.DataFrame(events).sort_values(["div_date", "ticker"]).reset_index(drop=True)
    q = (
        ev.groupby(["quarter", "quarter_end"], as_index=False)
        .agg(div_cash=("div_cash", "sum"))
        .sort_values("quarter_end")
        .reset_index(drop=True)
    )
    total = float(ev["div_cash"].sum())
    return {"total": total, "events": ev, "quarterly": q}


# =========================
# 5) Load data + reconcile
# =========================
portfolios_df = load_portfolios()
txns_all = load_txns()
baseline_all = load_baseline()

if not txns_all.empty:
    existing = set(portfolios_df["portfolio"].tolist())
    found = set(txns_all["portfolio"].astype(str).str.strip().tolist())
    missing = sorted([p for p in found if p and p not in existing])
    if missing:
        add = pd.DataFrame(
            [
                {
                    "portfolio": p,
                    "start_mode": "ledger_complete",
                    "as_of_date": pd.to_datetime(date.today()),
                    "starting_cash": 0.0,
                }
                for p in missing
            ]
        )
        portfolios_df = pd.concat([portfolios_df, add], ignore_index=True)
        save_portfolios(portfolios_df)

portfolio_names = portfolios_df["portfolio"].tolist()

# =========================
# 6) UI
# =========================
st.title("Brown Investment Group Portfolio")
st.caption("Public view is read-only. Admin can edit portfolios, baseline lots, and transactions.")

st.sidebar.header("Settings")
analyze_on_date = st.sidebar.date_input(
    "Analyze on date",
    value=date.today(),
    key="sidebar_analyze_on_date",
    help="All holdings, P&L, NAV, and contribution analytics are evaluated as of this date.",
)
match_method = st.sidebar.selectbox("Sell matching", ["FIFO", "LIFO"], index=0)

freq_choice = st.sidebar.selectbox("Chart frequency", ["Weekly (Mon)", "Daily"], index=0)
chart_freq = "W-MON" if freq_choice.startswith("Weekly") else "D"

# ---------- Public view ----------
st.subheader("Public View (read-only)")
public_tabs = st.tabs(["Overview"] + portfolio_names)


def render_public_portfolio(
    pname: str,
    analyze_date: date,
    nav_series: pd.Series | None = None,
    spy_px: pd.Series | None = None,
):
    meta = get_portfolio_meta(portfolios_df, pname)
    p_txns = txns_all[txns_all["portfolio"] == pname].copy()
    p_base = baseline_all[baseline_all["portfolio"] == pname].copy()

    snap = portfolio_snapshot(meta, p_txns, p_base, match_method, valuation_date=analyze_date)

    if snap["start_mode"] == "snapshot_start":
        st.info(
            f"Snapshot portfolio — tracking boundary starts {snap['as_of'].date()}. "
            f"Baseline lots represent holdings as of that date (long shares positive, short shares negative)."
        )
    else:
        st.caption("Ledger-complete portfolio — metrics reflect your ledger as entered.")

    st.caption(f"Analysis date: **{pd.to_datetime(analyze_date).date().isoformat()}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Starting Cash", f"${snap['starting_cash']:,.2f}")
    c2.metric("Cash", f"${snap['cash']:,.2f}")
    c3.metric("Market Value (signed)", f"${snap['market_value']:,.2f}")
    c4.metric("NAV", f"${snap['nav']:,.2f}")

    d1, d2 = st.columns(2)
    d1.metric("Unrealized P&L", f"${snap['unrealized_pnl']:,.2f}")
    d2.metric("Realized P&L", f"${snap['realized_pnl']:,.2f}")

    divpack = compute_dividend_accrual_quarterly(pname, meta, txns_all, baseline_all, pd.to_datetime(analyze_date))
    st.metric("Dividends (estimated, quarterly accrual)", f"${float(divpack['total']):,.2f}")

    st.divider()
    st.markdown("## Tier 1 Analytics")

    sector_alloc, industry_alloc = build_allocation_tables(snap)
    a1, a2 = st.columns([1, 1])
    with a1:
        st.markdown("### Sector allocation (ABS exposure incl. cash)")
        render_sector_pie(sector_alloc, f"{pname} — Sector Exposure (Abs)")
        st.dataframe(
            sector_alloc.assign(weight_pct=(sector_alloc["weight"] * 100).round(2)).drop(columns=["weight"]),
            use_container_width=True,
        )
    with a2:
        st.markdown("### Sector → Industry breakdown (ABS exposure, Yahoo Finance)")
        st.dataframe(
            industry_alloc.assign(weight_pct=(industry_alloc["weight"] * 100).round(2)).drop(columns=["weight"]),
            use_container_width=True,
        )

    st.markdown("### Contribution to return (P&L contribution by ticker)")
    contrib = build_contribution_table(snap)
    if contrib.empty:
        st.info("No contribution data yet (need holdings and/or sells/covers/dividends).")
    else:
        show_contrib = contrib.copy()
        for col in ["unrealized_return_pct", "realized_return_pct"]:
            show_contrib[col] = pd.to_numeric(show_contrib[col], errors="coerce")
            show_contrib[col] = show_contrib[col].map(lambda x: f"{x:,.2f}%" if pd.notna(x) else "N/A")
        st.dataframe(show_contrib, use_container_width=True)
        chart_df = contrib.set_index("ticker")[["total_contribution"]]
        st.bar_chart(chart_df)

    st.markdown("### Contribution to return — price breakdown by ticker")
    px_breakdown = build_price_breakdown_table(snap, valuation_date=analyze_date)
    px_breakdown = build_price_breakdown_table(snap)
    if px_breakdown.empty:
        st.info("No price breakdown available yet (need holdings and/or closed trades).")
    else:
        show_px = px_breakdown.copy()
        for col in ["avg_buy_price", "avg_sold_price", "current_price"]:
            show_px[col] = pd.to_numeric(show_px[col], errors="coerce")
            show_px[col] = show_px[col].map(lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A")
        st.dataframe(show_px, use_container_width=True)

    st.markdown("### Dividend tracker (estimated)")
    if divpack["quarterly"] is None or divpack["quarterly"].empty:
        st.info("No dividend events found for held LONG tickers in this period.")
    else:
        st.dataframe(divpack["quarterly"], use_container_width=True)
        st.bar_chart(divpack["quarterly"].set_index("quarter_end")[["div_cash"]])

    st.markdown("### Drawdown")
    if nav_series is None or nav_series.dropna().empty:
        st.info("No NAV series yet for drawdown.")
    else:
        dd = compute_drawdown(nav_series)
        st.line_chart(dd)

    st.markdown("### Beta / Alpha vs SPY")
    if nav_series is None or spy_px is None or nav_series.dropna().empty or spy_px.dropna().empty:
        st.info("Beta/alpha need both portfolio NAV series and benchmark series.")
    else:
        stats = compute_beta_alpha(nav_series, spy_px)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Beta", f"{stats['beta']:.2f}" if np.isfinite(stats["beta"]) else "—")
        k2.metric("Alpha (annualized)", f"{stats['alpha_ann']*100:.2f}%" if np.isfinite(stats["alpha_ann"]) else "—")
        k3.metric("R²", f"{stats['r2']:.2f}" if np.isfinite(stats["r2"]) else "—")
        k4.metric("Obs", f"{stats['n']}")

        roll = rolling_beta_alpha(nav_series, spy_px, window=26)
        if not roll.empty:
            r1, r2 = st.columns(2)
            with r1:
                st.caption("Rolling beta (26 periods)")
                st.line_chart(roll[["beta"]])
            with r2:
                st.caption("Rolling alpha (annualized, 26 periods)")
                st.line_chart(roll[["alpha_ann"]])

    st.divider()
    if not snap["holdings"].empty:
        st.markdown("### Holdings (net shares can be negative for shorts)")
        st.dataframe(snap["holdings"], use_container_width=True)

    if not snap["lots"].empty:
        st.markdown("### Open lots (baseline + trades) — LONG & SHORT")
        st.dataframe(snap["lots"].sort_values(["ticker", "buy_date"]), use_container_width=True)

    if not snap["realized"].empty:
        rv = snap["realized"].copy()
        rv["realized_return_%"] = np.where(
            rv["buy_price"] > 0,
            ((rv["sell_price"] / rv["buy_price"]) - 1.0) * 100.0,
            np.nan,
        )
        st.markdown("### Realized matches (per lot) — sells & covers")
        st.dataframe(rv.sort_values(["sell_date", "ticker"], ascending=False), use_container_width=True)

    st.markdown("### Transactions (filtered by boundary when snapshot)")
    if snap["filtered_txns"].empty:
        st.write("No transactions.")
    else:
        st.dataframe(snap["filtered_txns"], use_container_width=True)


# ----------------------------
# Overview tab
# ----------------------------
with public_tabs[0]:
    st.markdown("## Overview")

    end_date = pd.to_datetime(analyze_on_date)
    nav_series_map = {}
    cash_now_by_port = {}
    pnl_now_by_port = {}
    div_total_by_port = {}

    for p in portfolio_names:
        meta = get_portfolio_meta(portfolios_df, p)
        p_txns = txns_all[txns_all["portfolio"] == p].copy()
        p_base = baseline_all[baseline_all["portfolio"] == p].copy()

        snap = portfolio_snapshot(meta, p_txns, p_base, match_method, valuation_date=analyze_on_date)
        cash_now_by_port[p] = snap["cash"]
        pnl_now_by_port[p] = snap["realized_pnl"] + snap["unrealized_pnl"]

        nav_series_map[p] = compute_nav_series_for_portfolio(
            pname=p,
            meta=meta,
            txns_all=txns_all,
            baseline_all=baseline_all,
            end_date=end_date,
            chart_freq=chart_freq,
        )

        divpack = compute_dividend_accrual_quarterly(p, meta, txns_all, baseline_all, end_date)
        div_total_by_port[p] = float(divpack["total"]) if divpack else 0.0

    nav_df = pd.DataFrame(nav_series_map).sort_index()

    if nav_df.empty:
        agg_nav_last = 0.0
        agg_nav = pd.Series(dtype=float)
    else:
        agg_nav = nav_df.ffill().sum(axis=1)
        agg_nav_last = float(agg_nav.iloc[-1])

    total_cash = float(np.nansum(list(cash_now_by_port.values()))) if cash_now_by_port else 0.0
    total_pnl = float(np.nansum(list(pnl_now_by_port.values()))) if pnl_now_by_port else 0.0
    total_div = float(np.nansum(list(div_total_by_port.values()))) if div_total_by_port else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ALL Total Cash", f"${total_cash:,.2f}")
    c2.metric("ALL Total Gains (P&L)", f"${total_pnl:,.2f}")
    c3.metric("ALL NAV (Cash + MV)", f"${agg_nav_last:,.2f}")
    c4.metric("ALL Dividends (estimated)", f"${total_div:,.2f}")

    st.divider()

    earliest = None
    for s in nav_series_map.values():
        fv = s.first_valid_index()
        if fv is not None:
            earliest = fv if earliest is None else min(earliest, fv)

    spy = pd.Series(dtype=float)
    if earliest is None or nav_df.empty:
        st.info("Add at least one portfolio with data to see charts.")
    else:
        spy_px = fetch_price_history(["SPY"], earliest, end_date)
        if not spy_px.empty:
            spy_daily = spy_px["SPY"].copy()
            if chart_freq == "D":
                spy = spy_daily.reindex(nav_df.index).ffill()
            else:
                spy = spy_daily.reindex(pd.date_range(earliest, end_date, freq="D")).ffill()
                spy = spy.reindex(nav_df.index).ffill()
            spy = spy.astype(float)
        else:
            spy = pd.Series(index=nav_df.index, dtype=float)

        rel = pd.DataFrame({p: index_to_100(nav_df[p]) for p in nav_df.columns})
        rel["SPY"] = index_to_100(spy) if not spy.empty else spy

        st.markdown(f"### Relative performance (Indexed to 100 at series start) — {freq_choice}")
        st.line_chart(rel)

        if not agg_nav.empty:
            st.markdown(f"### Aggregate growth (Indexed) — {freq_choice}")
            st.line_chart(index_to_100(agg_nav))

        st.markdown("### Aggregate drawdown")
        dd_agg = compute_drawdown(agg_nav)
        if not dd_agg.empty:
            st.line_chart(dd_agg)

    st.divider()
    st.markdown("### Dividend totals by portfolio (estimated)")
    div_tbl = pd.DataFrame(
        [{"portfolio": p, "dividends_estimated": float(div_total_by_port.get(p, 0.0))} for p in portfolio_names]
    ).sort_values("dividends_estimated", ascending=False)
    st.dataframe(div_tbl, use_container_width=True)


# ----------------------------
# Individual portfolio tabs
# ----------------------------
for i, p in enumerate(portfolio_names, start=1):
    with public_tabs[i]:
        st.markdown(f"## {p}")

        nav = nav_series_map.get(p)
        spy_aligned = None
        if nav is not None and not nav.dropna().empty:
            s0 = nav.first_valid_index()
            s1 = nav.last_valid_index()
            if s0 is not None and s1 is not None:
                spy_px = fetch_price_history(["SPY"], s0, s1)
                if not spy_px.empty:
                    spy_daily = spy_px["SPY"].copy()
                    if chart_freq == "D":
                        spy_aligned = spy_daily.reindex(nav.index).ffill()
                    else:
                        spy_aligned = spy_daily.reindex(pd.date_range(s0, s1, freq="D")).ffill()
                        spy_aligned = spy_aligned.reindex(nav.index).ffill()

        render_public_portfolio(p, analyze_date=analyze_on_date, nav_series=nav, spy_px=spy_aligned)

# ---------- Admin section ----------
if is_admin:
    import io
    import zipfile

    def build_backup_zip() -> bytes:
        files = [
            ("portfolios.csv", PORTFOLIO_PATH),
            ("transactions.csv", TXN_PATH),
            ("baseline_lots.csv", BASELINE_PATH),
        ]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            for arcname, path in files:
                try:
                    with open(path, "rb") as f:
                        z.writestr(arcname, f.read())
                except FileNotFoundError:
                    z.writestr(arcname, b"")
        return buf.getvalue()

    st.markdown("---")
    st.header("Admin")

    # =========================
    # Create portfolio (TOP)
    # =========================
    st.subheader("Create portfolio")
    with st.form("create_portfolio_form", clear_on_submit=True):
        new_name = st.text_input("Name", placeholder="e.g., Long Only, Trading, IRA")
        new_mode = st.selectbox("Start mode", VALID_MODES, index=0)
        new_asof = st.date_input("As-of date", value=date.today(), min_value=PORTFOLIO_MIN_DATE)
        new_cash = st.number_input("Starting cash ($)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
        submitted = st.form_submit_button("Add portfolio")

    if submitted:
        name = _clean_str(new_name)
        if not name:
            st.error("Enter a portfolio name.")
        elif name in set(portfolio_names):
            st.warning("That portfolio already exists.")
        else:
            portfolios_df = pd.concat(
                [
                    portfolios_df,
                    pd.DataFrame(
                        [
                            {
                                "portfolio": name,
                                "start_mode": new_mode,
                                "as_of_date": pd.to_datetime(new_asof),
                                "starting_cash": float(new_cash),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            save_portfolios(portfolios_df)
            st.success("Portfolio added.")
            st.rerun()

    st.divider()

    # =========================
    # Active portfolio selector
    # =========================
    active = st.selectbox("Active portfolio", portfolio_names, index=0, key="admin_active_portfolio")
    meta = get_portfolio_meta(portfolios_df, active)

    # =========================
    # Tabs
    # =========================
    admin_tabs = st.tabs(["Transactions", "Portfolio Settings", "Baseline lots", "Documentation"])

    # -----------------------
    # Transactions tab
    # -----------------------
    with admin_tabs[0]:
        st.subheader("Transactions")

        price_key = f"txn_price__{active}"
        ticker_key = f"txn_ticker__{active}"
        shares_key = f"txn_shares__{active}"
        use_rec_key = f"use_recommended_price__{active}"

        refresh_flag_key = f"refresh_price_flag__{active}"
        refresh_value_key = f"refresh_price_value__{active}"

        if st.session_state.get(refresh_flag_key, False):
            v = st.session_state.get(refresh_value_key, None)
            if v is not None:
                st.session_state[price_key] = float(v)
            st.session_state[refresh_flag_key] = False

        refresh_pressed = False
        add_txn = False

        # -----------------------
        # Add transaction (manual)
        # -----------------------
        with st.form("add_txn_form", clear_on_submit=False):
            txn_type = st.selectbox("Type", ["buy", "sell", "short", "cover", "dividend"], index=0)
            t_date = st.date_input("Date", value=date.today())

            if txn_type in ["buy", "sell", "short", "cover"]:
                st.caption("Shorting flow: **short** opens/increases a short; **cover** closes/reduces a short.")
                c1, c2, c3 = st.columns([1, 1, 1])

                with c1:
                    st.text_input("Ticker", value="AAPL", key=ticker_key)
                with c2:
                    st.number_input(
                        "Shares",
                        min_value=0.0,
                        value=1.000,
                        step=0.0001,
                        format="%.4f",
                        key=shares_key,
                    )

                t_amount = np.nan

                tkr_clean = str(st.session_state.get(ticker_key, "")).upper().strip()
                chosen_date = t_date

                if use_rec_key not in st.session_state:
                    st.session_state[use_rec_key] = True
                st.checkbox("Use recommended close price", key=use_rec_key)

                rec = fetch_close_on_or_before(tkr_clean, chosen_date) if tkr_clean else None

                if price_key not in st.session_state:
                    st.session_state[price_key] = float(rec) if (st.session_state[use_rec_key] and rec is not None) else 100.00

                if st.session_state[use_rec_key] and rec is not None:
                    st.session_state[price_key] = float(rec)

                with c3:
                    st.number_input(
                        "Price",
                        min_value=0.0,
                        value=float(st.session_state[price_key]),
                        step=0.0001,
                        format="%.4f",
                        key=price_key,
                        help="Auto-fills with close on/before the transaction date (editable).",
                    )
                    refresh_pressed = st.form_submit_button("🔄 Refresh price")

                if rec is not None and tkr_clean:
                    st.caption(f"Recommended close for {tkr_clean} on/before {chosen_date.isoformat()}: **${rec:,.4f}**")

            else:
                c1, c2 = st.columns([1, 1])
                with c1:
                    t_ticker = st.text_input("Ticker (optional)", value="")
                with c2:
                    t_amount = st.number_input(
                        "Dividend amount ($)", min_value=0.0, value=0.0, step=1.0, format="%.2f"
                    )

            add_txn = st.form_submit_button("Save transaction")

        if refresh_pressed:
            tkr_clean = str(st.session_state.get(ticker_key, "")).upper().strip()
            chosen_date = t_date
            if not tkr_clean:
                st.warning("Enter a ticker first.")
            else:
                rec2 = fetch_close_on_or_before(tkr_clean, chosen_date)
                if rec2 is None:
                    st.error("Could not fetch a close price for that ticker/date.")
                else:
                    st.session_state[refresh_value_key] = float(rec2)
                    st.session_state[refresh_flag_key] = True
                    st.rerun()

        if add_txn:
            txn_type_now = txn_type

            if txn_type_now in ["buy", "sell", "short", "cover"]:
                t_ticker_final = str(st.session_state.get(ticker_key, "")).upper().strip()
                t_shares_final = float(st.session_state.get(shares_key, 0.0))
                t_price_final = float(st.session_state.get(price_key, 0.0))
                t_amount_final = np.nan
            else:
                t_ticker_final = _clean_str(t_ticker).upper()
                t_shares_final = np.nan
                t_price_final = np.nan
                t_amount_final = float(t_amount) if pd.notna(t_amount) else np.nan

            row = {
                "txn_id": str(pd.Timestamp.utcnow().value),
                "portfolio": active,
                "date": pd.to_datetime(t_date),
                "type": txn_type_now,
                "ticker": t_ticker_final,
                "shares": float(round(t_shares_final, 4)) if pd.notna(t_shares_final) else np.nan,
                "price": float(t_price_final) if pd.notna(t_price_final) else np.nan,
                "amount": float(t_amount_final) if pd.notna(t_amount_final) else np.nan,
            }

            candidate_txns = pd.concat([txns_all, pd.DataFrame([row])], ignore_index=True)
            p_txns = candidate_txns[candidate_txns["portfolio"] == active].copy()
            p_base = baseline_all[baseline_all["portfolio"] == active].copy()

            ok, msg = validate_candidate_state(meta, p_txns, p_base, match_method)
            if not ok:
                st.error(msg)
            else:
                txns_all = candidate_txns
                save_txns(txns_all)
                st.success("Saved.")
                st.rerun()

        # -----------------------
        # Bulk upload (beneath Add transaction)
        # -----------------------
        st.divider()
        st.subheader("Bulk upload (CSV)")

        up = st.file_uploader(
            "Upload transactions CSV ❓",
            type=["csv"],
            key=f"bulk_upload_csv__{active}",
            help="""
Required CSV headers (exact):

TICKER | TRANSACTION TYPE | DATE | SHARE COUNT

Optional:
PRICE

• TICKER: AAPL, SPY (upper/lower ok)
• TRANSACTION TYPE: buy, sell, short, cover (anything else ignored)
• DATE: YYYY-MM-DD recommended
• SHARE COUNT: positive number (parentheses allowed, e.g. (30))
• PRICE (optional): if blank/missing, the app will fetch Yahoo close on/before DATE

Notes:
• Import can Append or Replace the selected portfolio
• Invalid rows are skipped and shown
""",
        )

        mode = st.radio(
            "Import mode for this portfolio",
            ["Append", "Replace"],
            horizontal=True,
            key=f"bulk_mode__{active}",
        )

        preview_key_good = f"bulk_good__{active}"
        preview_key_bad = f"bulk_bad__{active}"
        preview_key_raw = f"bulk_raw__{active}"
        file_sig_key = f"bulk_file_sig__{active}"

        def _file_signature(f) -> str:
            name = getattr(f, "name", "unknown")
            size = getattr(f, "size", "unknown")
            return f"{name}::{size}"

        # IMPORTANT: only re-parse + clear previews when the file changes
        if up is not None:
            sig = _file_signature(up)
            prev_sig = st.session_state.get(file_sig_key)
            if sig != prev_sig:
                st.session_state[file_sig_key] = sig
                try:
                    raw_import = parse_simple_txn_csv(up)
                    st.session_state[preview_key_raw] = raw_import
                    st.session_state.pop(preview_key_good, None)
                    st.session_state.pop(preview_key_bad, None)
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    st.session_state[preview_key_raw] = pd.DataFrame()
                    st.session_state.pop(preview_key_good, None)
                    st.session_state.pop(preview_key_bad, None)

        raw_import = st.session_state.get(preview_key_raw, pd.DataFrame())

        if isinstance(raw_import, pd.DataFrame) and not raw_import.empty:
            st.write("Parsed rows (pre-price fill):")
            st.dataframe(raw_import, use_container_width=True)

            if st.button("Preview with prices", key=f"bulk_preview_btn__{active}"):
                good, bad = enrich_import_with_prices(raw_import)
                st.session_state[preview_key_good] = good
                st.session_state[preview_key_bad] = bad
                st.rerun()

        good = st.session_state.get(preview_key_good, pd.DataFrame())
        bad = st.session_state.get(preview_key_bad, pd.DataFrame())

        if isinstance(bad, pd.DataFrame) and not bad.empty:
            st.warning(f"Skipped: {len(bad)} rows.")
            st.dataframe(bad, use_container_width=True)

        if isinstance(good, pd.DataFrame) and not good.empty:
            st.success(f"Ready to import: {len(good)} rows.")
            st.dataframe(good, use_container_width=True)

            if st.button("✅ Import into portfolio", type="primary", key=f"bulk_import_btn__{active}"):
                rows = []
                base_id = pd.Timestamp.utcnow().value
                for i, r in good.reset_index(drop=True).iterrows():
                    rows.append(
                        {
                            "txn_id": str(base_id + i),
                            "portfolio": active,
                            "date": pd.to_datetime(r["date"]),
                            "type": str(r["type"]).lower(),
                            "ticker": str(r["ticker"]).upper().strip(),
                            "shares": float(round(float(r["shares"]), 3)),
                            "price": float(r["price"]),
                            "amount": np.nan,
                        }
                    )

                imported_df = pd.DataFrame(rows)

                if mode == "Replace":
                    candidate_txns = txns_all[txns_all["portfolio"] != active].copy()
                    candidate_txns = pd.concat([candidate_txns, imported_df], ignore_index=True)
                else:
                    candidate_txns = pd.concat([txns_all, imported_df], ignore_index=True)

                p_txns2 = candidate_txns[candidate_txns["portfolio"] == active].copy()
                p_base2 = baseline_all[baseline_all["portfolio"] == active].copy()

                ok, msg = validate_candidate_state(meta, p_txns2, p_base2, match_method)
                if not ok:
                    st.error(f"Import rejected: {msg}")
                else:
                    txns_all = candidate_txns
                    save_txns(txns_all)
                    st.success(f"Imported {len(imported_df)} transactions into '{active}' ({mode}).")

                    st.session_state.pop(preview_key_good, None)
                    st.session_state.pop(preview_key_bad, None)
                    st.session_state.pop(preview_key_raw, None)
                    st.rerun()
        elif up is not None and isinstance(raw_import, pd.DataFrame) and raw_import.empty:
            st.info("No valid buy/sell/short/cover rows found in this file (or required columns missing).")

        # -----------------------
        # Delete transaction
        # -----------------------
        st.divider()
        st.subheader("Delete transaction")

        p_txns = txns_all[txns_all["portfolio"] == active].copy()
        if p_txns.empty:
            st.info("No transactions.")
        else:
            disp = p_txns.copy()
            disp["date_str"] = pd.to_datetime(disp["date"]).dt.date.astype(str)

            trade_mask = disp["type"].isin(["buy", "sell", "short", "cover"])
            disp.loc[trade_mask, "desc"] = (
                disp.loc[trade_mask, "date_str"]
                + " | "
                + disp.loc[trade_mask, "type"]
                + " | "
                + disp.loc[trade_mask, "ticker"]
                + " | "
                + disp.loc[trade_mask, "shares"].map(lambda x: f"{x:.3f}")
                + " @ "
                + disp.loc[trade_mask, "price"].map(lambda x: f"{x:.2f}")
            )
            disp.loc[~trade_mask, "desc"] = (
                disp.loc[~trade_mask, "date_str"]
                + " | dividend | "
                + disp.loc[~trade_mask, "ticker"].fillna("").astype(str)
                + " | $"
                + disp.loc[~trade_mask, "amount"].map(lambda x: f"{x:.2f}")
            )

            disp["display"] = disp["desc"] + " | id=" + disp["txn_id"].astype(str)

            choice = st.selectbox("Select transaction", disp["display"].tolist(), key="del_txn_choice")
            chosen_id = choice.split("id=")[-1].strip()

            if st.button("Delete selected transaction"):
                candidate_txns = txns_all[txns_all["txn_id"].astype(str) != chosen_id].copy()

                p_txns2 = candidate_txns[candidate_txns["portfolio"] == active].copy()
                p_base = baseline_all[baseline_all["portfolio"] == active].copy()
                ok, msg = validate_candidate_state(meta, p_txns2, p_base, match_method)

                if not ok:
                    st.error(f"Delete rejected: {msg}")
                else:
                    txns_all = candidate_txns
                    save_txns(txns_all)
                    st.warning("Deleted.")
                    st.rerun()

    # -----------------------
    # Portfolio Settings tab
    # -----------------------
    with admin_tabs[1]:
        st.subheader(f"Portfolio settings: {active}")

        with st.form("edit_portfolio_form"):
            mode_u = st.selectbox("Start mode", VALID_MODES, index=VALID_MODES.index(meta["start_mode"]))
            asof_u = st.date_input("As-of date", value=meta["as_of_date"].date(), min_value=PORTFOLIO_MIN_DATE)
            cash_u = st.number_input(
                "Starting cash ($)",
                min_value=0.0,
                value=float(meta["starting_cash"]),
                step=100.0,
                format="%.2f",
            )
            saved = st.form_submit_button("Save settings")

        if saved:
            portfolios_df.loc[portfolios_df["portfolio"] == active, "start_mode"] = mode_u
            portfolios_df.loc[portfolios_df["portfolio"] == active, "as_of_date"] = pd.to_datetime(asof_u)
            portfolios_df.loc[portfolios_df["portfolio"] == active, "starting_cash"] = float(cash_u)
            save_portfolios(portfolios_df)
            st.success("Saved.")
            st.rerun()

    # -----------------------
    # Baseline lots tab
    # -----------------------
    with admin_tabs[2]:
        st.subheader("Baseline lots (snapshot portfolios only)")

        if meta["start_mode"] != "snapshot_start":
            st.info("This portfolio is ledger-complete. Baseline lots are only used for snapshot-start portfolios.")
        else:
            with st.form("add_baseline_form", clear_on_submit=True):
                c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
                with c1:
                    bl_ticker = st.text_input("Ticker", value="AAPL")
                with c2:
                    bl_side = st.selectbox("Side", ["LONG", "SHORT"], index=0)
                with c3:
                    bl_shares = st.number_input("Shares", min_value=0.0, value=1.000, step=0.001, format="%.3f")
                with c4:
                    bl_price = st.number_input(
                        "Entry / cost basis (price)", min_value=0.0, value=100.00, step=0.01, format="%.2f"
                    )
                with c5:
                    bl_date = st.date_input("Acquisition date", value=date.today())

                add_bl = st.form_submit_button("Add baseline lot")

            if add_bl:
                signed_sh = float(round(bl_shares, 3))
                if bl_side == "SHORT":
                    signed_sh = -signed_sh

                row = {
                    "lot_id": str(pd.Timestamp.utcnow().value),
                    "portfolio": active,
                    "ticker": _clean_str(bl_ticker).upper(),
                    "buy_date": pd.to_datetime(bl_date),
                    "buy_price": float(bl_price),
                    "shares_open": signed_sh,
                }
                baseline_all = pd.concat([baseline_all, pd.DataFrame([row])], ignore_index=True)
                save_baseline(baseline_all)
                st.success("Baseline lot added.")
                st.rerun()

            st.divider()
            st.write("Current baseline lots (negative shares = short)")
            p_base = baseline_all[baseline_all["portfolio"] == active].copy()

            if p_base.empty:
                st.info("No baseline lots yet.")
            else:
                st.dataframe(p_base.sort_values(["ticker", "buy_date"]), use_container_width=True)

                p_base_disp = p_base.copy()
                p_base_disp["display"] = (
                    pd.to_datetime(p_base_disp["buy_date"]).dt.date.astype(str)
                    + " | "
                    + p_base_disp["ticker"]
                    + " | "
                    + p_base_disp["shares_open"].map(lambda x: f"{x:.3f}")
                    + " @ "
                    + p_base_disp["buy_price"].map(lambda x: f"{x:.2f}")
                    + " | id="
                    + p_base_disp["lot_id"].astype(str)
                )
                choice = st.selectbox("Select baseline lot to delete", p_base_disp["display"].tolist())
                chosen_id = choice.split("id=")[-1].strip()

                if st.button("Delete baseline lot"):
                    baseline_all = baseline_all[baseline_all["lot_id"].astype(str) != chosen_id].copy()
                    save_baseline(baseline_all)
                    st.warning("Deleted baseline lot.")
                    st.rerun()

    # -----------------------
    # Documentation tab
    # -----------------------
    with admin_tabs[3]:
        st.subheader("Documentation")
        st.markdown(
            """
### Quick start

**If you have full history (ledger-complete):**
1. Go to **Portfolio Settings** → set **Start mode** = `ledger_complete`
2. Set **Starting cash** (cash at the beginning of your ledger)
3. Enter **all trades/dividends** in **Transactions**

**If you only have today's holdings (snapshot-start):**
1. Go to **Portfolio Settings** → set **Start mode** = `snapshot_start`
2. Set **As-of date** = the snapshot boundary (usually today)
3. Set **Starting cash** = cash in the account on the as-of date
4. Go to **Baseline lots** → add each holding:
   - LONG positions: Side=LONG, Shares positive
   - SHORT positions: Side=SHORT, Shares positive (stored as negative internally)
   - Entry / cost basis price
   - Acquisition date
5. Enter only **new** transactions after the as-of date in **Transactions**

### Shorting

- Use **short** to open / increase a short position (cash increases).
- Use **cover** to buy-to-cover (cash decreases).
- **sell** is only for reducing LONG shares.
- **cover** is only for reducing SHORT shares.
- This app enforces **no negative cash** (no margin loans). If a transaction would make cash go negative, it is rejected.

**Dividends:** estimated dividend accrual is calculated on LONG shares only (does not model borrow costs / dividends-in-lieu on shorts).

### Bulk upload CSV

Required headers:
- TICKER
- TRANSACTION TYPE
- DATE
- SHARE COUNT

Optional:
- PRICE  (if blank/missing, we fetch Yahoo close on/before DATE)
"""
        )

    # =========================
    # Backup download
    # =========================
    st.divider()
    st.subheader("Backup")
    st.caption("Download a ZIP containing: portfolios.csv, transactions.csv, baseline_lots.csv")

    st.download_button(
        "⬇️ Download CSV backup (ZIP)",
        data=build_backup_zip(),
        file_name=f"portfolio_backup_{date.today().isoformat()}.zip",
        mime="application/zip",
    )
