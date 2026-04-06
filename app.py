# app.py
import hmac
import logging
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

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

    with st.sidebar.expander("Admin access", expanded=False):
        st.caption("Lower-priority controls")
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

# =========================
# 2) Storage (CSV)
# =========================
TXN_PATH = "transactions.csv"
PORTFOLIO_PATH = "portfolios.csv"
BASELINE_PATH = "baseline_lots.csv"

TXN_COLS = ["txn_id", "portfolio", "date", "type", "ticker", "shares", "price", "amount"]
PORTFOLIO_COLS = ["portfolio", "start_mode", "as_of_date", "starting_cash", "credit_spread_pct"]
VALID_MODES = ["ledger_complete", "snapshot_start"]
BASELINE_COLS = ["lot_id", "portfolio", "ticker", "buy_date", "buy_price", "shares_open"]

# Transaction types:
# - buy: open/increase long
# - sell: reduce/close long
# - short: open/increase short (borrow+sell)
# - cover: buy-to-cover reduce/close short
# - dividend: cash dividend
# - credit_interest: cash sweep / monthly interest credit
VALID_TXN_TYPES = ["buy", "sell", "short", "cover", "dividend", "credit_interest"]
PORTFOLIO_MIN_DATE = date(2000, 1, 1)

# Monthly auto-credit model:
# interest for a completed month
# = beginning_of_month_cash * max(^IRX - haircut, 0) / 12
DEFAULT_CREDIT_SPREAD_PCT = 0.50
IRX_PROXY_TICKER = "^IRX"

SECTOR_BUCKETS = [
    "Infrastructure",
    "Real Estate",
    "Technology",
    "Media & Telecommunications",
    "Consumer & Retail",
    "Healthcare",
    "Natural Resources & Energy",
    "Industrials",
    "Financial Institutions",
]

SECTOR_BUCKET_MAP = {
    "real estate": "Real Estate",
    "technology": "Technology",
    "communication services": "Media & Telecommunications",
    "consumer cyclical": "Consumer & Retail",
    "consumer defensive": "Consumer & Retail",
    "healthcare": "Healthcare",
    "basic materials": "Natural Resources & Energy",
    "energy": "Natural Resources & Energy",
    "industrials": "Industrials",
    "financial services": "Financial Institutions",
    "utilities": "Infrastructure",
}


def classify_sector_bucket(sector: str, industry: str) -> str:
    s = str(sector or "").strip().lower()
    i = str(industry or "").strip().lower()

    if s in SECTOR_BUCKET_MAP:
        return SECTOR_BUCKET_MAP[s]

    if any(k in i for k in ["bank", "insurance", "asset management", "credit", "financial"]):
        return "Financial Institutions"
    if any(k in i for k in ["telecom", "media", "broadcast", "entertainment", "advertising"]):
        return "Media & Telecommunications"
    if any(k in i for k in ["reit", "property", "real estate"]):
        return "Real Estate"
    if any(k in i for k in ["software", "semiconductor", "internet", "it services", "technology"]):
        return "Technology"
    if any(k in i for k in ["biotech", "pharma", "drug", "medical", "health"]):
        return "Healthcare"
    if any(k in i for k in ["oil", "gas", "mining", "metals", "chemical", "energy"]):
        return "Natural Resources & Energy"
    if any(k in i for k in ["airline", "aerospace", "construction", "machinery", "industrial", "transport"]):
        return "Industrials"
    if any(k in i for k in ["consumer", "retail", "restaurant", "apparel", "auto", "food", "beverage"]):
        return "Consumer & Retail"
    if any(k in i for k in ["utility", "power", "water", "infrastructure"]):
        return "Infrastructure"
    return "Unknown"


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
    df["credit_spread_pct"] = (
        pd.to_numeric(df["credit_spread_pct"], errors="coerce").fillna(DEFAULT_CREDIT_SPREAD_PCT)
    )

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
                            "credit_spread_pct": DEFAULT_CREDIT_SPREAD_PCT,
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
    out["credit_spread_pct"] = (
        pd.to_numeric(out["credit_spread_pct"], errors="coerce").fillna(DEFAULT_CREDIT_SPREAD_PCT)
    )
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

    income_rows = df[~is_trade].dropna(subset=["amount"]).copy()
    income_rows = income_rows[income_rows["amount"] >= 0]
    income_rows["ticker"] = income_rows["ticker"].fillna("").astype(str)

    out = pd.concat([trades, income_rows], ignore_index=True)
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

    if data is None or data.empty:
        return pd.Series(dtype=float)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = pd.DataFrame({tickers[0]: data["Close"]})

    close.index = pd.to_datetime(close.index, errors="coerce")
    close = close[~close.index.isna()].sort_index()
    close = close.ffill()
    close = close.dropna(axis=1, how="all")
    if close.empty:
        return pd.Series(dtype=float)

    close = close.iloc[-1]

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
    t = str(ticker).upper().strip()
    s = pd.to_datetime(start).normalize()
    e = pd.to_datetime(end).normalize()

    if not t:
        return pd.Series(dtype=float, name=t)

    def _normalize_dividends(div_like: pd.Series | pd.DataFrame | None) -> pd.Series:
        if div_like is None:
            return pd.Series(dtype=float, name=t)

        if isinstance(div_like, pd.DataFrame):
            if "Dividends" in div_like.columns:
                div_s = div_like["Dividends"]
            elif div_like.shape[1] == 1:
                div_s = div_like.iloc[:, 0]
            else:
                return pd.Series(dtype=float, name=t)
        else:
            div_s = div_like

        if div_s is None or len(div_s) == 0:
            return pd.Series(dtype=float, name=t)

        idx = pd.to_datetime(div_s.index)
        try:
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert(None)
        except Exception:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                pass

        idx = pd.to_datetime(idx).normalize()
        out = pd.Series(pd.to_numeric(div_s.values, errors="coerce"), index=idx, name=t)
        out = out.dropna().groupby(level=0).sum().sort_index()
        out = out[(out.index >= s) & (out.index <= e)].copy()
        out = out[out > 0]
        return out.astype(float)

    # Yahoo endpoints can intermittently return an empty `.dividends` series.
    # Try a few sources in priority order and use the first non-empty result.
    providers: list[pd.Series] = []

    try:
        providers.append(_normalize_dividends(yf.Ticker(t).dividends))
    except Exception:
        pass

    try:
        hist_actions = yf.Ticker(t).history(period="max", auto_adjust=False, actions=True)
        providers.append(_normalize_dividends(hist_actions))
    except Exception:
        pass

    try:
        dl = yf.download(
            [t],
            start=s.date(),
            end=(e + pd.Timedelta(days=1)).date(),
            interval="1d",
            auto_adjust=False,
            actions=True,
            progress=False,
        )
        if isinstance(dl.columns, pd.MultiIndex):
            if ("Dividends", t) in dl.columns:
                providers.append(_normalize_dividends(dl[("Dividends", t)]))
            elif "Dividends" in dl.columns.get_level_values(0):
                providers.append(_normalize_dividends(dl["Dividends"]))
        else:
            providers.append(_normalize_dividends(dl.get("Dividends")))
    except Exception:
        pass

    for candidate in providers:
        if candidate is not None and not candidate.empty:
            return candidate

    return pd.Series(dtype=float, name=t)


@st.cache_data(ttl=3600)
def fetch_latest_irx_rate_pct() -> float | None:
    try:
        data = yf.download(IRX_PROXY_TICKER, period="5d", interval="1d", auto_adjust=False, progress=False)
        if data is None or data.empty:
            return None

        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            if IRX_PROXY_TICKER not in close.columns:
                return None
            s = close[IRX_PROXY_TICKER]
        else:
            s = close

        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return None

        return float(s.iloc[-1])
    except Exception:
        return None


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
      PRICE

    Returns normalized df:
      ticker, type, date, shares, price
    Only keeps trade type in {buy, sell, short, cover}.
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


def parse_credit_interest_csv(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Required columns:
      MONTH | AMOUNT

    Optional columns:
      TICKER

    Returns (good, bad):
      good columns: date, amount, ticker
      bad includes original values + reason
    """
    df = pd.read_csv(uploaded_file)
    df.columns = [str(c).strip().upper() for c in df.columns]

    required = ["MONTH", "AMOUNT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = pd.DataFrame()
    work["month_raw"] = df["MONTH"].astype(str).str.strip()
    work["amount_raw"] = df["AMOUNT"]
    work["ticker"] = (
        df["TICKER"].astype(str).str.upper().str.strip() if "TICKER" in df.columns else ""
    )

    month_str = work["month_raw"].astype(str).str.replace(r"[/.]", "-", regex=True)
    parsed_month = pd.to_datetime(month_str, errors="coerce")
    month_is_yyyy_mm = month_str.str.match(r"^\d{4}-\d{2}$")
    parsed_month = parsed_month.where(~month_is_yyyy_mm, pd.to_datetime(month_str + "-01", errors="coerce"))

    work["date"] = parsed_month.dt.to_period("M").dt.to_timestamp()
    work["amount"] = pd.to_numeric(work["amount_raw"].apply(_parse_number_allow_parens), errors="coerce")

    bad_reasons = []
    for _, r in work.iterrows():
        if pd.isna(r["date"]):
            bad_reasons.append("Invalid MONTH (use YYYY-MM)")
        elif pd.isna(r["amount"]):
            bad_reasons.append("Invalid AMOUNT")
        elif float(r["amount"]) < 0:
            bad_reasons.append("AMOUNT must be >= 0")
        else:
            bad_reasons.append("")

    work["reason"] = bad_reasons
    good = work.loc[work["reason"] == "", ["date", "amount", "ticker"]].copy()
    bad = work.loc[work["reason"] != "", ["month_raw", "amount_raw", "ticker", "reason"]].copy()
    bad = bad.rename(columns={"month_raw": "MONTH", "amount_raw": "AMOUNT", "ticker": "TICKER"})

    good["amount"] = pd.to_numeric(good["amount"], errors="coerce")
    good["ticker"] = good["ticker"].fillna("").astype(str)
    good = good.dropna(subset=["date", "amount"])
    good = good[good["amount"] >= 0].reset_index(drop=True)
    bad = bad.reset_index(drop=True)
    return good, bad


# =========================
# 3) Lot engine + accounting (LONG + SHORT)
# =========================
def apply_reduce_to_lots(lots: list[dict], reduce_shares: float, method: str, side: str):
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

        else:
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
    if typ in ["dividend", "credit_interest"]:
        return +float(row["amount"])
    return 0.0


def get_portfolio_meta(portfolios_df: pd.DataFrame, portfolio: str):
    r = portfolios_df.loc[portfolios_df["portfolio"] == portfolio].iloc[0]
    return {
        "portfolio": str(portfolio),
        "start_mode": str(r["start_mode"]),
        "as_of_date": pd.to_datetime(r["as_of_date"]),
        "starting_cash": float(r["starting_cash"]),
        "credit_spread_pct": float(r.get("credit_spread_pct", DEFAULT_CREDIT_SPREAD_PCT)),
    }


def _cash_start_boundary(meta: dict, txns: pd.DataFrame) -> pd.Timestamp:
    as_of = pd.to_datetime(meta["as_of_date"]).normalize()

    if meta["start_mode"] == "snapshot_start":
        return as_of

    if txns is None or txns.empty:
        return as_of

    d = pd.to_datetime(txns["date"], errors="coerce").dropna()
    if d.empty:
        return as_of

    return min(as_of, d.min().normalize())


def build_auto_credit_interest_txns(
    portfolio_meta: dict,
    portfolio_txns_all: pd.DataFrame,
    valuation_date: date | None = None,
) -> pd.DataFrame:
    eval_date = pd.to_datetime(valuation_date if valuation_date is not None else date.today()).normalize()
    txns = portfolio_txns_all.copy()

    if txns.empty:
        txns = pd.DataFrame(columns=TXN_COLS)
    else:
        txns["date"] = pd.to_datetime(txns["date"], errors="coerce")
        txns = txns.dropna(subset=["date"]).copy()
        txns = txns[txns["date"] <= eval_date].copy()

    start_boundary = _cash_start_boundary(portfolio_meta, txns)

    if portfolio_meta["start_mode"] == "snapshot_start" and not txns.empty:
        txns = txns[txns["date"] >= pd.to_datetime(portfolio_meta["as_of_date"]).normalize()].copy()

    if eval_date < start_boundary:
        return pd.DataFrame(columns=TXN_COLS)

    months = pd.period_range(start=start_boundary.to_period("M"), end=eval_date.to_period("M"), freq="M")
    months = [m for m in months if m.end_time.normalize() <= eval_date]

    if not months:
        return pd.DataFrame(columns=TXN_COLS)

    if txns.empty:
        work = pd.DataFrame(columns=TXN_COLS + ["month"])
    else:
        work = txns.copy()
        work["month"] = pd.to_datetime(work["date"]).dt.to_period("M")

    manual_credit_months = set()
    if not work.empty:
        manual_credit_months = set(work.loc[work["type"] == "credit_interest", "month"].tolist())

    running_cash = float(portfolio_meta["starting_cash"])
    auto_rows = []

    irx_rate_pct = fetch_latest_irx_rate_pct()
    if irx_rate_pct is None or not np.isfinite(irx_rate_pct):
        return pd.DataFrame(columns=TXN_COLS)

    haircut_pct = float(portfolio_meta.get("credit_spread_pct", DEFAULT_CREDIT_SPREAD_PCT))
    implied_annual_rate_pct = max(irx_rate_pct - haircut_pct, 0.0)

    for month in months:
        month_txns = (
            work[work["month"] == month].sort_values(["date", "txn_id"])
            if not work.empty
            else pd.DataFrame()
        )
        month_start_cash = running_cash
        month_had_auto = False

        if month not in manual_credit_months and month_start_cash > 1e-12:
            interest_amt = round(month_start_cash * (implied_annual_rate_pct / 100.0) / 12.0, 2)

            if interest_amt > 0:
                month_end = month.end_time.normalize()
                auto_rows.append(
                    {
                        "txn_id": f"auto_credit_interest__{portfolio_meta['portfolio']}__{month.strftime('%Y%m')}",
                        "portfolio": portfolio_meta["portfolio"],
                        "date": month_end,
                        "type": "credit_interest",
                        "ticker": "",
                        "shares": np.nan,
                        "price": np.nan,
                        "amount": float(interest_amt),
                    }
                )
                month_had_auto = True

        if month_txns is not None and not month_txns.empty:
            running_cash += float(month_txns.apply(cash_delta, axis=1).sum())

        if month_had_auto:
            running_cash += float(auto_rows[-1]["amount"])

    if not auto_rows:
        return pd.DataFrame(columns=TXN_COLS)

    auto_df = pd.DataFrame(auto_rows, columns=TXN_COLS)
    auto_df["date"] = pd.to_datetime(auto_df["date"])
    return auto_df


def build_effective_txns(
    portfolio_meta: dict,
    portfolio_txns_all: pd.DataFrame,
    valuation_date: date | None = None,
) -> pd.DataFrame:
    base = portfolio_txns_all.copy()
    if base.empty:
        base = pd.DataFrame(columns=TXN_COLS)
    else:
        base["date"] = pd.to_datetime(base["date"], errors="coerce")

    auto_df = build_auto_credit_interest_txns(
        portfolio_meta=portfolio_meta,
        portfolio_txns_all=base,
        valuation_date=valuation_date,
    )

    out = pd.concat([base, auto_df], ignore_index=True)
    if out.empty:
        return out

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "type"]).copy()
    out = out.sort_values(["date", "txn_id"]).reset_index(drop=True)
    return out


def build_monthly_credit_income_report(effective_txns: pd.DataFrame) -> pd.DataFrame:
    if effective_txns is None or effective_txns.empty:
        return pd.DataFrame(columns=["month", "manual_credit_income", "auto_credit_income", "total_credit_income"])

    tx = effective_txns.copy()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
    tx = tx.dropna(subset=["date"])
    tx = tx[tx["type"] == "credit_interest"].copy()
    if tx.empty:
        return pd.DataFrame(columns=["month", "manual_credit_income", "auto_credit_income", "total_credit_income"])

    tx["month"] = tx["date"].dt.to_period("M").dt.to_timestamp()
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    tx["source"] = np.where(tx["txn_id"].astype(str).str.startswith("auto_credit_interest__"), "auto", "manual")

    grouped = tx.groupby(["month", "source"], as_index=False)["amount"].sum()
    pivot = grouped.pivot(index="month", columns="source", values="amount").fillna(0.0)
    pivot = pivot.rename(columns={"manual": "manual_credit_income", "auto": "auto_credit_income"})
    for c in ["manual_credit_income", "auto_credit_income"]:
        if c not in pivot.columns:
            pivot[c] = 0.0
    pivot["total_credit_income"] = pivot["manual_credit_income"] + pivot["auto_credit_income"]
    out = pivot.reset_index().sort_values("month", ascending=False)
    return out[["month", "manual_credit_income", "auto_credit_income", "total_credit_income"]]


def validate_candidate_state(
    portfolio_meta: dict,
    portfolio_txns_all: pd.DataFrame,
    portfolio_baseline: pd.DataFrame,
    method: str,
) -> tuple[bool, str]:
    txns = portfolio_txns_all.copy()
    if not txns.empty:
        txns["date"] = pd.to_datetime(txns["date"], errors="coerce")
        txns = txns.dropna(subset=["date"]).copy()

    start_mode = portfolio_meta["start_mode"]
    as_of = pd.to_datetime(portfolio_meta["as_of_date"]).normalize()

    if start_mode == "snapshot_start" and not txns.empty:
        if (txns["date"] < as_of).any():
            return False, f"Snapshot portfolios cannot have transactions before as-of date ({as_of.date()})."

    validation_date = date.today()
    if not txns.empty:
        validation_date = pd.to_datetime(txns["date"]).max().date()

    effective_txns = build_effective_txns(
        portfolio_meta=portfolio_meta,
        portfolio_txns_all=txns,
        valuation_date=validation_date,
    )

    cash = float(portfolio_meta["starting_cash"])
    if not effective_txns.empty:
        for _, r in effective_txns.sort_values(["date", "txn_id"]).iterrows():
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
                return False, f"Invalid SELL: not enough LONG shares of {t} to sell {sh:.4f}."
            long_shares[t] -= sh

        elif typ == "short":
            short_shares[t] = short_shares.get(t, 0.0) + sh

        elif typ == "cover":
            if short_shares.get(t, 0.0) + 1e-12 < sh:
                return False, f"Invalid COVER: not enough SHORT shares of {t} to cover {sh:.4f}."
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

    df = build_effective_txns(
        portfolio_meta=portfolio_meta,
        portfolio_txns_all=txns,
        valuation_date=eval_date.date(),
    )

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
    chart_freq: str,
) -> pd.Series:
    start = _portfolio_start_date(meta)
    end = pd.to_datetime(end_date).normalize()

    base_txns = txns_all[txns_all["portfolio"] == pname].copy()
    if not base_txns.empty:
        base_txns["date"] = pd.to_datetime(base_txns["date"], errors="coerce").dt.normalize()
        base_txns = base_txns.dropna(subset=["date"]).copy()

    # For ledger-complete portfolios, chart history should begin at the first available
    # economic activity, not strictly the configured as-of date.
    if str(meta.get("start_mode", "")).strip().lower() == "ledger_complete":
        start_candidates: list[pd.Timestamp] = [start]
        if not base_txns.empty:
            start_candidates.append(base_txns["date"].min())

        baseline_port = baseline_all[baseline_all["portfolio"] == pname].copy()
        if not baseline_port.empty and "buy_date" in baseline_port.columns:
            baseline_port["buy_date"] = pd.to_datetime(baseline_port["buy_date"], errors="coerce").dt.normalize()
            baseline_port = baseline_port.dropna(subset=["buy_date"])
            if not baseline_port.empty:
                start_candidates.append(baseline_port["buy_date"].min())

        start = min(start_candidates)

    if start > end:
        start = end

    txns = build_effective_txns(
        portfolio_meta=meta,
        portfolio_txns_all=base_txns,
        valuation_date=end.date(),
    )

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
    df["ticker_display"] = np.where(df["label"] == "CASH", "CASH", df["ticker"])
    df["sector_bucket"] = [classify_sector_bucket(s, i) for s, i in zip(df["sector"], df["industry"]) ]
    df.loc[df["label"] == "CASH", "sector_bucket"] = "Cash"

    sector_alloc = (
        df.groupby(["sector_bucket"], as_index=False)
        .agg(exposure=("exposure", "sum"))
        .rename(columns={"sector_bucket": "sector"})
        .sort_values("exposure", ascending=False)
    )
    sector_total = float(sector_alloc["exposure"].sum())
    sector_alloc["weight"] = np.where(sector_total > 0, sector_alloc["exposure"] / sector_total, 0.0)

    industry_alloc = (
        df.groupby(["sector_bucket", "industry"], as_index=False)
        .agg(
            exposure=("exposure", "sum"),
            tickers=(
                "ticker_display",
                lambda s: ", ".join(sorted({str(t).strip() for t in s if str(t).strip()})),
            ),
        )
        .rename(columns={"sector_bucket": "sector"})
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

    cash_income = pd.DataFrame(columns=["ticker", "dividend_pnl"])
    tx = snap["filtered_txns"]
    if tx is not None and not tx.empty:
        d = tx[tx["type"].isin(["dividend", "credit_interest"])].copy()
        if not d.empty:
            d["ticker"] = d["ticker"].fillna("").astype(str).str.upper().str.strip()
            d.loc[d["ticker"] == "", "ticker"] = "CASH"
            cash_income = d.groupby("ticker", as_index=False).agg(dividend_pnl=("amount", "sum"))

    out = unreal.merge(realized, on="ticker", how="outer").merge(cash_income, on="ticker", how="outer")
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
        buy_avg["avg_buy_price"] = np.where(
            buy_avg["total_shares"] > 0,
            buy_avg["weighted"] / buy_avg["total_shares"],
            np.nan,
        )
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

    tickers = set(buy_avg.get("ticker", pd.Series(dtype=str)).tolist()) | set(
        sold_avg.get("ticker", pd.Series(dtype=str)).tolist()
    )
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
    out["current_price"] = out["ticker"].map(live)

    return out.sort_values("ticker").reset_index(drop=True)


def render_sector_pie(sector_alloc: pd.DataFrame, title: str):
    plot_df = sector_alloc.copy()
    plot_df["exposure"] = pd.to_numeric(plot_df["exposure"], errors="coerce").fillna(0.0)
    plot_df = plot_df[plot_df["exposure"] > 1e-12].copy()

    if plot_df.empty or plot_df["exposure"].sum() <= 0:
        st.info("No sector allocation available yet.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        plot_df["exposure"],
        labels=plot_df["sector"],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        labeldistance=1.14,
        textprops={"fontsize": 9},
    )
    ax.set_title(title, pad=24)
    fig.subplots_adjust(top=0.80)
    ax.axis("equal")
    st.pyplot(fig, clear_figure=True)


# =========================
# 4c) Dividend tracker (manual-posted first, market-estimated fallback)
# =========================
def compute_daily_shares_df(
    pname: str,
    meta: dict,
    txns_all: pd.DataFrame,
    baseline_all: pd.DataFrame,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
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
    start = _portfolio_start_date(meta)
    end = pd.to_datetime(end_date).normalize()

    manual = txns_all[txns_all["portfolio"] == pname].copy()
    if not manual.empty:
        manual["date"] = pd.to_datetime(manual["date"], errors="coerce").dt.normalize()
        manual["type"] = manual["type"].astype(str).str.lower().str.strip()
        manual["ticker"] = manual["ticker"].fillna("").astype(str).str.upper().str.strip()
        manual["amount"] = pd.to_numeric(manual["amount"], errors="coerce").fillna(0.0)
        manual = manual[
            (manual["type"] == "dividend")
            & manual["date"].notna()
            & (manual["date"] >= start)
            & (manual["date"] <= end)
            & (manual["amount"] > 0)
        ].copy()
    if not manual.empty:
        manual["ticker"] = manual["ticker"].replace("", "CASH")
        manual["div_date"] = manual["date"]
        manual["div_per_share"] = np.nan
        manual["shares"] = np.nan
        manual["div_cash"] = manual["amount"]
        manual["quarter"] = manual["div_date"].dt.to_period("Q").astype(str)
        manual["quarter_end"] = manual["div_date"].dt.to_period("Q").dt.end_time.dt.normalize()
        manual["source"] = "manual_posted"
        cols = ["ticker", "div_date", "div_per_share", "shares", "div_cash", "quarter", "quarter_end", "source"]
        ev_manual = manual[cols].sort_values(["div_date", "ticker"]).reset_index(drop=True)
        q_manual = (
            ev_manual.groupby(["quarter", "quarter_end"], as_index=False)
            .agg(div_cash=("div_cash", "sum"))
            .sort_values("quarter_end")
            .reset_index(drop=True)
        )
        return {
            "total": float(ev_manual["div_cash"].sum()),
            "events": ev_manual,
            "quarterly": q_manual,
            "diagnostics": {"used_manual": True, "long_tickers_checked": [], "no_dividend_data_tickers": []},
        }

    shares_df = compute_daily_shares_df(pname, meta, txns_all, baseline_all, end)
    if shares_df.empty or shares_df.shape[1] == 0:
        return {
            "total": 0.0,
            "events": pd.DataFrame(),
            "quarterly": pd.DataFrame(),
            "diagnostics": {"used_manual": False, "long_tickers_checked": [], "no_dividend_data_tickers": []},
        }

    tickers = list(shares_df.columns)
    events = []
    long_tickers_checked = []
    no_dividend_data_tickers = []

    for t in tickers:
        max_long = float(np.nanmax(pd.to_numeric(shares_df[t], errors="coerce").fillna(0.0).values)) if t in shares_df.columns else 0.0
        if max_long <= 1e-12:
            continue
        long_tickers_checked.append(t)

        div = fetch_dividends_series(t, start, end)
        if div is None or div.empty:
            no_dividend_data_tickers.append(t)
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
                    "source": "market_estimated",
                }
            )

    if not events:
        return {
            "total": 0.0,
            "events": pd.DataFrame(),
            "quarterly": pd.DataFrame(),
            "diagnostics": {
                "used_manual": False,
                "long_tickers_checked": sorted(long_tickers_checked),
                "no_dividend_data_tickers": sorted(no_dividend_data_tickers),
            },
        }

    ev = pd.DataFrame(events).sort_values(["div_date", "ticker"]).reset_index(drop=True)
    q = (
        ev.groupby(["quarter", "quarter_end"], as_index=False)
        .agg(div_cash=("div_cash", "sum"))
        .sort_values("quarter_end")
        .reset_index(drop=True)
    )
    total = float(ev["div_cash"].sum())
    return {
        "total": total,
        "events": ev,
        "quarterly": q,
        "diagnostics": {
            "used_manual": False,
            "long_tickers_checked": sorted(long_tickers_checked),
            "no_dividend_data_tickers": sorted(no_dividend_data_tickers),
        },
    }


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
                    "credit_spread_pct": DEFAULT_CREDIT_SPREAD_PCT,
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
st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 1520px;
            padding-top: 1.25rem;
            padding-bottom: 1.25rem;
        }
        .appbar {
            border: 1px solid rgba(99, 115, 129, 0.35);
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(20,25,32,0.95), rgba(16,20,26,0.95));
            color: #F3F6FA;
            padding: 12px 16px;
            margin-bottom: 16px;
        }
        .appbar-grid {
            display: grid;
            grid-template-columns: 1.7fr 1.3fr;
            gap: 16px;
            align-items: center;
        }
        .appbar-title { font-size: 1.05rem; font-weight: 700; margin: 0; }
        .appbar-sub { font-size: 0.78rem; color: #9AA4B2; margin: 2px 0 0 0; }
        .status-badge {
            display: inline-block;
            font-size: 0.72rem;
            border: 1px solid rgba(73, 206, 255, 0.5);
            color: #7FDBFF;
            border-radius: 999px;
            padding: 2px 8px;
            margin-left: 8px;
        }
        .filter-bar {
            position: sticky;
            top: 0.5rem;
            z-index: 10;
            background: rgba(11,16,23,0.95);
            border: 1px solid rgba(99,115,129,0.3);
            border-radius: 12px;
            color: #F3F6FA;
            padding: 8px 12px;
            margin-bottom: 16px;
        }
        .kpi-section-title {
            color: #9AA4B2;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 14px 0 6px;
            font-weight: 600;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4,minmax(0,1fr));
            gap: 12px;
            margin: 0 0 14px;
        }
        .kpi-card {
            border: 1px solid rgba(99,115,129,0.35);
            border-radius: 12px;
            padding: 12px;
            background: rgba(21,27,36,0.92);
            color: #F3F6FA;
        }
        .kpi-label { font-size: 0.74rem; color: #9AA4B2; text-transform: uppercase; letter-spacing: 0.05em; }
        .kpi-value { font-size: 1.35rem; font-weight: 700; margin: 6px 0; }
        .kpi-delta-pos { color: #37C48B; font-size: 0.82rem; }
        .kpi-delta-neg { color: #FF6B6B; font-size: 0.82rem; }
        .kpi-note { color: #9AA4B2; font-size: 0.75rem; margin-top: 4px; }
        @media (max-width: 1200px) {
            .kpi-grid { grid-template-columns: repeat(3,minmax(0,1fr)); }
        }
        @media (max-width: 900px) {
            .kpi-grid { grid-template-columns: repeat(2,minmax(0,1fr)); }
        }
        @media (max-width: 620px) {
            .kpi-grid { grid-template-columns: 1fr; }
        }
        .page-title { font-size: 1.5rem; font-weight: 700; margin: 4px 0 0; }
        .page-subtitle { color: #9AA4B2; margin-top: 2px; }
        .review-box {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.75rem;
            padding: 0.75rem 0.95rem;
            background: rgba(30, 38, 52, 0.45);
            color: #F3F6FA;
            margin-bottom: 0.8rem;
        }
        .review-box h4 {
            margin: 0 0 0.4rem 0;
            font-size: 1rem;
        }
        .review-box ul {
            margin: 0;
            padding-left: 1.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fmt_delta(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v*100:+.2f}%"


def render_kpi_cards(items: list[dict]) -> None:
    cards = []
    for item in items:
        delta = item.get("delta")
        delta_label = item.get("delta_label")
        delta_line = ""
        if delta_label and delta is not None and pd.notna(delta):
            delta_class = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
            delta_line = f"<div class='{delta_class}'>{delta_label}: {_fmt_delta(delta)}</div>"

        cards.append(
            (
                "<div class='kpi-card'>"
                f"<div class='kpi-label'>{item['label']}</div>"
                f"<div class='kpi-value'>{item['value']}</div>"
                f"{delta_line}"
                f"<div class='kpi-note'>{item.get('note', '')}</div>"
                "</div>"
            )
        )
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_kpi_sections(sections: list[dict]) -> None:
    for section in sections:
        st.markdown(f"<div class='kpi-section-title'>{section['title']}</div>", unsafe_allow_html=True)
        render_kpi_cards(section["items"])

with st.sidebar:
    st.header("Navigation")
    nav_focus = st.radio("Primary area", ["Overview", "Portfolios", "Admin"], horizontal=True)
    with st.expander("Filters", expanded=True):
        analyze_on_date = st.date_input(
            "Analyze on date",
            value=date.today(),
            key="sidebar_analyze_on_date",
            help="All holdings, P&L, NAV, and contribution analytics are evaluated as of this date.",
        )
        match_method = st.selectbox("Sell matching", ["FIFO", "LIFO"], index=0)
        freq_choice = st.selectbox("Chart frequency", ["Weekly (Mon)", "Daily"], index=0)
    with st.expander("Admin", expanded=False):
        st.write("Mode:", "✅ Admin (edit enabled)" if is_admin else "👀 Public (read-only)")
        if is_admin and st.button("Log out"):
            st.session_state.is_admin = False
            st.rerun()

chart_freq = "W-MON" if freq_choice.startswith("Weekly") else "D"

st.markdown(
    f"""
    <div class='appbar'>
        <div class='appbar-grid'>
            <div>
                <p class='appbar-title'>🏛️ Brown Investment Group Portfolio Platform <span class='status-badge'>{'ADMIN' if is_admin else 'PUBLIC'}</span></p>
                <p class='appbar-sub'>Institutional multi-portfolio analytics • Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            <div style='text-align:right;color:#9AA4B2;'>
                ⤓ Export&nbsp;&nbsp;&nbsp;↗ Share&nbsp;&nbsp;&nbsp;⚙ Settings&nbsp;&nbsp;&nbsp;👤 User
            </div>
        </div>
    </div>
    <div class='filter-bar'><b>Sticky filters:</b> Portfolio scope: {nav_focus} &nbsp;|&nbsp; Analyze date: {analyze_on_date} &nbsp;|&nbsp; Frequency: {freq_choice} &nbsp;|&nbsp; Benchmark: SPY</div>
    """,
    unsafe_allow_html=True,
)

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
    credit_income_report = build_monthly_credit_income_report(snap["filtered_txns"])

    st.markdown(f"<div class='page-title'>{pname}</div><div class='page-subtitle'>Portfolio drill-down workspace for performance, attribution, income, risk, and positions.</div>", unsafe_allow_html=True)
    if snap["start_mode"] == "snapshot_start":
        st.info(
            f"Snapshot portfolio — tracking boundary starts {snap['as_of'].date()}. "
            f"Baseline lots represent holdings as of that date (long shares positive, short shares negative)."
        )
    else:
        st.caption("Ledger-complete portfolio — metrics reflect your ledger as entered.")

    with st.expander("How to read this portfolio", expanded=False):
        st.markdown(
            """
            <div class="review-box">
                <ul>
                    <li><b>Health:</b> Start with value, cash, and liquidity metrics.</li>
                    <li><b>Performance:</b> Compare NAV trajectory vs benchmark and drawdown.</li>
                    <li><b>Drivers:</b> Use contribution tables for sector and position effects.</li>
                    <li><b>Risk:</b> Review beta, concentration, volatility proxies, and exposures.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _period_return(nav_s: pd.Series | None, lookback_days: int | None = None) -> float:
        if nav_s is None:
            return np.nan
        s = nav_s.dropna().sort_index()
        if s.empty:
            return np.nan

        end_val = float(s.iloc[-1])
        if abs(end_val) < 1e-12:
            return np.nan

        if lookback_days is None:
            start_val = float(s.iloc[0])
        else:
            cutoff = s.index.max() - pd.Timedelta(days=lookback_days)
            hist = s[s.index <= cutoff]
            if hist.empty:
                return np.nan
            start_val = float(hist.iloc[-1])

        if abs(start_val) < 1e-12:
            return np.nan
        return (end_val / start_val) - 1.0

    st.caption(f"Analysis date: **{pd.to_datetime(analyze_date).date().isoformat()}**")

    credit_income_total = float(credit_income_report["total_credit_income"].sum()) if not credit_income_report.empty else 0.0
    divpack = compute_dividend_accrual_quarterly(pname, meta, txns_all, baseline_all, pd.to_datetime(analyze_date))

    total_return = _period_return(nav_series, None)
    one_year_return = _period_return(nav_series, 365)
    quarterly_return = _period_return(nav_series, 90)

    nav_change_1p = np.nan
    if nav_series is not None:
        nav_points = nav_series.dropna().sort_index()
        if len(nav_points) >= 2:
            prev_nav = float(nav_points.iloc[-2])
            curr_nav = float(nav_points.iloc[-1])
            if abs(prev_nav) > 1e-12:
                nav_change_1p = (curr_nav / prev_nav) - 1.0

    current_dd = np.nan
    if nav_series is not None and not nav_series.dropna().empty:
        dd_now = compute_drawdown(nav_series).dropna()
        if not dd_now.empty:
            current_dd = float(dd_now.iloc[-1])
    kpi_sections = [
        {
            "title": "Portfolio health",
            "items": [
                {"label": "Portfolio Value", "value": f"${float(snap['nav']):,.0f}", "delta": nav_change_1p, "delta_label": "vs previous period", "note": "Net asset value"},
                {"label": "Cash", "value": f"${float(snap['cash']):,.0f}", "note": "Available liquidity"},
                {"label": "Drawdown", "value": f"{current_dd*100:,.2f}%" if pd.notna(current_dd) else "N/A", "note": "From peak NAV"},
                {"label": "Unrealized Gain", "value": f"${float(snap['unrealized_pnl']):,.0f}", "note": "Open positions"},
            ],
        },
        {
            "title": "Performance",
            "items": [
                {"label": "Total Return", "value": f"{total_return*100:,.2f}%" if pd.notna(total_return) else "N/A", "note": "Since inception"},
                {"label": "1Y Return", "value": f"{one_year_return*100:,.2f}%" if pd.notna(one_year_return) else "N/A", "note": "Trailing 12 months (requires >=1 year of NAV history)"},
                {"label": "Quarterly Return", "value": f"{quarterly_return*100:,.2f}%" if pd.notna(quarterly_return) else "N/A", "note": "Trailing quarter (requires >=90 days of NAV history)"},
                {"label": "Realized Gain", "value": f"${float(snap['realized_pnl']):,.0f}", "note": "Closed positions"},
            ],
        },
        {
            "title": "Income",
            "items": [
                {"label": "Dividends", "value": f"${float(divpack['total']):,.0f}", "note": "Estimated accrual"},
                {"label": "Credit/Margin", "value": f"${credit_income_total:,.0f}", "note": "Interest income"},
            ],
        },
    ]
    render_kpi_sections(kpi_sections)

    nav_obs = 0 if nav_series is None else int(nav_series.dropna().shape[0])
    if nav_obs < 2:
        st.info(
            "Performance charts need at least two NAV observations. "
            "With only one valuation point, return and drawdown panels will be blank or N/A."
        )
    if divpack["quarterly"] is None or divpack["quarterly"].empty:
        st.info(
            "Dividend tracker only shows estimated cash when BOTH conditions are met: "
            "(1) the portfolio had LONG shares on dividend dates, and "
            "(2) the ticker has dividend events available from market data."
        )

    section_tabs = st.tabs(["Performance", "Attribution", "Income", "Risk", "Positions"])

    with section_tabs[0]:
        st.markdown("### NAV / performance")
        if nav_series is None or nav_series.dropna().empty:
            st.info("No NAV series yet for performance.")
        else:
            perf_nav = nav_series.dropna().sort_index().to_frame(name="NAV")
            st.line_chart(perf_nav)
        cperf1, cperf2 = st.columns(2)
        with cperf1:
            st.markdown("### Drawdown")
            if nav_series is None or nav_series.dropna().empty:
                st.info("No NAV series yet for drawdown.")
            else:
                dd = compute_drawdown(nav_series)
                st.line_chart(dd)
        with cperf2:
            st.markdown("### Rolling / monthly returns")
            if nav_series is None or nav_series.dropna().empty:
                st.info("No return series available.")
            else:
                ret = nav_series.pct_change().dropna()
                if ret.empty:
                    st.info("Need at least two NAV points to compute period returns.")
                else:
                    st.bar_chart(ret.tail(24))

    with section_tabs[1]:
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
            st.markdown("### Sector → Industry breakdown (ABS exposure)")
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
        if px_breakdown.empty:
            st.info("No price breakdown available yet (need holdings and/or closed trades).")
        else:
            show_px = px_breakdown.copy()
            for col in ["avg_buy_price", "avg_sold_price", "current_price"]:
                show_px[col] = pd.to_numeric(show_px[col], errors="coerce")
                show_px[col] = show_px[col].map(lambda x: f"${x:,.4f}" if pd.notna(x) else "N/A")
            st.dataframe(show_px, use_container_width=True)

    with section_tabs[2]:
        st.markdown("### Monthly credit income")
        if credit_income_report.empty:
            st.info("No monthly credit interest income posted yet.")
        else:
            show_credit = credit_income_report.copy()
            show_credit["month"] = pd.to_datetime(show_credit["month"]).dt.strftime("%Y-%m")
            st.dataframe(show_credit, use_container_width=True)
            chart_credit = credit_income_report.copy()
            chart_credit["month"] = pd.to_datetime(chart_credit["month"])
            st.bar_chart(chart_credit.set_index("month")[["total_credit_income"]].sort_index())

        st.markdown("### Dividend tracker (posted or estimated)")
        if divpack["quarterly"] is None or divpack["quarterly"].empty:
            st.info(
                "No dividend events were matched for held LONG tickers in this analysis window. "
                "If holdings were short-only, cash-only, or very recent, this can be expected."
            )
            diag = divpack.get("diagnostics", {})
            checked = diag.get("long_tickers_checked", []) if isinstance(diag, dict) else []
            no_data = diag.get("no_dividend_data_tickers", []) if isinstance(diag, dict) else []
            if checked and len(no_data) == len(checked):
                st.warning(
                    "Market dividend data was empty for all held LONG tickers in this window: "
                    + ", ".join(no_data)
                    + ". This usually means the upstream Yahoo endpoint returned no actions data."
                )
        else:
            st.dataframe(divpack["quarterly"], use_container_width=True)
            st.bar_chart(divpack["quarterly"].set_index("quarter_end")[["div_cash"]])

    with section_tabs[3]:
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

    with section_tabs[4]:
        if not snap["holdings"].empty:
            st.markdown("### Holdings (searchable, sortable)")
            h = snap["holdings"].copy()
            rename_map = {
                "ticker": "ticker",
                "net_shares": "shares",
                "avg_cost": "cost basis",
                "last_price": "price",
                "market_value": "market value",
                "unrealized_pnl": "unrealized P/L",
                "realized_pnl": "realized P/L",
                "dividend_yield": "dividend yield",
                "sector_bucket": "sector",
            }
            h = h.rename(columns={k: v for k, v in rename_map.items() if k in h.columns})
            query = st.text_input("Search ticker or name", key=f"holding_query_{pname}")
            quick = st.selectbox("Quick filter", ["All", "Winners", "Losers", "Largest Positions", "Highest Yield"], key=f"quick_hold_{pname}")
            if query:
                qq = query.lower()
                mask = h.get("ticker", pd.Series("", index=h.index)).astype(str).str.lower().str.contains(qq)
                if "name" in h.columns:
                    mask = mask | h["name"].astype(str).str.lower().str.contains(qq)
                h = h[mask]
            if quick == "Winners" and "unrealized P/L" in h.columns:
                h = h[h["unrealized P/L"] > 0]
            elif quick == "Losers" and "unrealized P/L" in h.columns:
                h = h[h["unrealized P/L"] < 0]
            elif quick == "Largest Positions" and "market value" in h.columns:
                h = h.sort_values("market value", ascending=False).head(15)
            elif quick == "Highest Yield" and "dividend yield" in h.columns:
                h = h.sort_values("dividend yield", ascending=False).head(15)
            if "last updated" not in h.columns:
                h["last updated"] = pd.to_datetime(analyze_date).date().isoformat()
            st.dataframe(h, use_container_width=True)

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
    c4.metric("ALL Dividends (posted/estimated)", f"${total_div:,.2f}")

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
    st.markdown("### Dividend totals by portfolio (posted/estimated)")
    div_tbl = pd.DataFrame(
        [{"portfolio": p, "dividends_total": float(div_total_by_port.get(p, 0.0))} for p in portfolio_names]
    ).sort_values("dividends_total", ascending=False)
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

    st.subheader("Create portfolio")
    with st.form("create_portfolio_form", clear_on_submit=True):
        new_name = st.text_input("Name", placeholder="e.g., Long Only, Trading, IRA")
        new_mode = st.selectbox("Start mode", VALID_MODES, index=0)
        new_asof = st.date_input("As-of date", value=date.today(), min_value=PORTFOLIO_MIN_DATE)
        new_cash = st.number_input("Starting cash ($)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
        new_credit_spread = st.number_input(
            "IRX haircut / spread (% annual)",
            value=DEFAULT_CREDIT_SPREAD_PCT,
            step=0.05,
            format="%.2f",
        )
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
                                "credit_spread_pct": float(new_credit_spread),
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

    active = st.selectbox("Active portfolio", portfolio_names, index=0, key="admin_active_portfolio")
    meta = get_portfolio_meta(portfolios_df, active)

    admin_tabs = st.tabs(["Transactions", "Portfolio Settings", "Baseline lots", "Documentation"])

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

        with st.form("add_txn_form", clear_on_submit=False):
            txn_type = st.selectbox("Type", ["buy", "sell", "short", "cover", "dividend", "credit_interest"], index=0)
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
                    st.session_state[price_key] = (
                        float(rec) if (st.session_state[use_rec_key] and rec is not None) else 100.00
                    )

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
                st.caption(
                    "Non-trade cash income rows supported: "
                    "**dividend** and **credit_interest**. "
                    "Auto monthly credit_interest is generated using beginning-of-month cash and latest ^IRX less your configured haircut."
                )
                c1, c2 = st.columns([1, 1])
                with c1:
                    t_ticker = st.text_input("Ticker (optional)", value="")
                with c2:
                    label = "Cash income amount ($)" if txn_type == "credit_interest" else "Dividend amount ($)"
                    t_amount = st.number_input(label, min_value=0.0, value=0.0, step=1.0, format="%.2f")

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

        st.divider()
        st.subheader("Monthly credit_interest upload (CSV)")
        credit_up = st.file_uploader(
            "Upload monthly credit_interest CSV",
            type=["csv"],
            key=f"credit_upload_csv__{active}",
            help="""
Required CSV headers (exact):

MONTH | AMOUNT

Optional:
TICKER

Examples:
• MONTH: 2024-01
• AMOUNT: 125.50
• TICKER (optional): CASH

Notes:
• This uploader is only for monthly credit_interest rows.
• MONTH is saved as the first day of the month.
• Import can Append or Replace existing credit_interest rows for this portfolio.
""",
        )

        credit_mode = st.radio(
            "Credit interest import mode",
            ["Append", "Replace credit_interest"],
            horizontal=True,
            key=f"credit_mode__{active}",
        )

        credit_good_key = f"credit_good__{active}"
        credit_bad_key = f"credit_bad__{active}"
        credit_file_sig_key = f"credit_file_sig__{active}"

        if credit_up is not None:
            credit_sig = _file_signature(credit_up)
            prev_credit_sig = st.session_state.get(credit_file_sig_key)
            if credit_sig != prev_credit_sig:
                st.session_state[credit_file_sig_key] = credit_sig
                try:
                    credit_good, credit_bad = parse_credit_interest_csv(credit_up)
                    st.session_state[credit_good_key] = credit_good
                    st.session_state[credit_bad_key] = credit_bad
                except Exception as e:
                    st.error(f"Credit-interest upload failed: {e}")
                    st.session_state[credit_good_key] = pd.DataFrame()
                    st.session_state[credit_bad_key] = pd.DataFrame()

        credit_good = st.session_state.get(credit_good_key, pd.DataFrame())
        credit_bad = st.session_state.get(credit_bad_key, pd.DataFrame())

        if isinstance(credit_bad, pd.DataFrame) and not credit_bad.empty:
            st.warning(f"Skipped credit_interest rows: {len(credit_bad)}")
            st.dataframe(credit_bad, use_container_width=True)

        if isinstance(credit_good, pd.DataFrame) and not credit_good.empty:
            st.success(f"Ready to import credit_interest rows: {len(credit_good)}")

            credit_preview = credit_good.copy()
            credit_preview["date"] = pd.to_datetime(credit_preview["date"]).dt.date.astype(str)
            st.dataframe(credit_preview, use_container_width=True)

            if st.button("✅ Import monthly credit_interest", type="primary", key=f"credit_import_btn__{active}"):
                rows = []
                base_id = pd.Timestamp.utcnow().value
                for i, r in credit_good.reset_index(drop=True).iterrows():
                    rows.append(
                        {
                            "txn_id": str(base_id + i),
                            "portfolio": active,
                            "date": pd.to_datetime(r["date"]),
                            "type": "credit_interest",
                            "ticker": str(r.get("ticker", "")).upper().strip(),
                            "shares": np.nan,
                            "price": np.nan,
                            "amount": float(r["amount"]),
                        }
                    )

                credit_import_df = pd.DataFrame(rows)
                if credit_mode == "Replace credit_interest":
                    keep = txns_all[~((txns_all["portfolio"] == active) & (txns_all["type"] == "credit_interest"))].copy()
                    candidate_txns = pd.concat([keep, credit_import_df], ignore_index=True)
                else:
                    candidate_txns = pd.concat([txns_all, credit_import_df], ignore_index=True)

                p_txns2 = candidate_txns[candidate_txns["portfolio"] == active].copy()
                p_base2 = baseline_all[baseline_all["portfolio"] == active].copy()

                ok, msg = validate_candidate_state(meta, p_txns2, p_base2, match_method)
                if not ok:
                    st.error(f"Credit-interest import rejected: {msg}")
                else:
                    txns_all = candidate_txns
                    save_txns(txns_all)
                    st.success(
                        f"Imported {len(credit_import_df)} credit_interest rows into '{active}' ({credit_mode})."
                    )
                    st.session_state.pop(credit_good_key, None)
                    st.session_state.pop(credit_bad_key, None)
                    st.rerun()
        elif credit_up is not None:
            st.info("No valid monthly credit_interest rows found in this file.")

        st.divider()
        st.subheader("Delete transaction")

        p_txns = txns_all[txns_all["portfolio"] == active].copy()
        if p_txns.empty:
            st.info("No transactions.")
        else:
            disp = p_txns.copy()
            disp["date_str"] = pd.to_datetime(disp["date"]).dt.date.astype(str)

            trade_mask = disp["type"].isin(["buy", "sell", "short", "cover"])

            trade_rows = disp.loc[trade_mask]
            trade_desc = pd.DataFrame(index=trade_rows.index)
            trade_desc["date_str"] = trade_rows["date_str"].astype("string[python]").fillna("")
            trade_desc["type"] = trade_rows["type"].astype("string[python]").fillna("")
            trade_desc["ticker"] = trade_rows["ticker"].astype("string[python]").fillna("")
            trade_desc["shares"] = (
                pd.to_numeric(trade_rows["shares"], errors="coerce")
                .fillna(0.0)
                .map(lambda x: f"{x:.3f}")
                .astype("string[python]")
            )
            trade_desc["price"] = (
                pd.to_numeric(trade_rows["price"], errors="coerce")
                .fillna(0.0)
                .map(lambda x: f"{x:.2f}")
                .astype("string[python]")
            )
            disp.loc[trade_mask, "desc"] = (
                trade_desc["date_str"]
                .str.cat(trade_desc["type"], sep=" | ")
                .str.cat(trade_desc["ticker"], sep=" | ")
                .str.cat(trade_desc["shares"], sep=" | ")
                .str.cat(trade_desc["price"], sep=" @ ")
            )

            cash_rows = disp.loc[~trade_mask]
            cash_desc = pd.DataFrame(index=cash_rows.index)
            cash_desc["date_str"] = cash_rows["date_str"].astype("string[python]").fillna("")
            cash_desc["type"] = cash_rows["type"].astype("string[python]").fillna("")
            cash_desc["ticker"] = cash_rows["ticker"].astype("string[python]").fillna("")
            cash_desc["amount"] = (
                pd.to_numeric(cash_rows["amount"], errors="coerce")
                .fillna(0.0)
                .map(lambda x: f"${x:.2f}")
                .astype("string[python]")
            )
            disp.loc[~trade_mask, "desc"] = (
                cash_desc["date_str"]
                .str.cat(cash_desc["type"], sep=" | ")
                .str.cat(cash_desc["ticker"], sep=" | ")
                .str.cat(cash_desc["amount"], sep=" | ")
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
            spread_u = st.number_input(
                "IRX haircut / spread (% annual)",
                value=float(meta.get("credit_spread_pct", DEFAULT_CREDIT_SPREAD_PCT)),
                step=0.05,
                format="%.2f",
            )
            saved = st.form_submit_button("Save settings")

        if saved:
            portfolios_df.loc[portfolios_df["portfolio"] == active, "start_mode"] = mode_u
            portfolios_df.loc[portfolios_df["portfolio"] == active, "as_of_date"] = pd.to_datetime(asof_u)
            portfolios_df.loc[portfolios_df["portfolio"] == active, "starting_cash"] = float(cash_u)
            portfolios_df.loc[portfolios_df["portfolio"] == active, "credit_spread_pct"] = float(spread_u)
            save_portfolios(portfolios_df)
            st.success("Saved.")
            st.rerun()

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

    with admin_tabs[3]:
        st.subheader("Documentation")
        st.markdown(
            """
### Quick start

**If you have full history (ledger-complete):**
1. Go to **Portfolio Settings** → set **Start mode** = `ledger_complete`
2. Set **Starting cash** (cash at the beginning of your ledger)
3. Enter **all trades/dividends/manual credit_interest** in **Transactions**

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

### Credit interest

- `credit_interest` is supported as a real cash-income transaction type.
- The app also auto-generates monthly `credit_interest` for completed months using:
  - beginning-of-month cash
  - latest **^IRX** as a proxy for cash yield
  - minus a configurable annual haircut/spread (default **0.50%**) set in Portfolio Settings
- If a manual `credit_interest` transaction already exists in a month, auto-accrual is skipped for that month.

**Dividends:** posted `dividend` transactions are used when available; otherwise an estimated accrual is calculated on LONG shares only (does not model borrow costs / dividends-in-lieu on shorts).

### Bulk upload CSV

Required headers:
- TICKER
- TRANSACTION TYPE
- DATE
- SHARE COUNT

Optional:
- PRICE  (if blank/missing, we fetch Yahoo close on/before DATE)

### Monthly credit interest CSV upload

Use the dedicated uploader in **Transactions** for manual monthly credit interest.

Required headers:
- MONTH (e.g. `2024-01`)
- AMOUNT

Optional:
- TICKER

This creates `credit_interest` transactions dated on the first day of each month.
"""
        )

    st.divider()
    st.subheader("Backup")
    st.caption("Download a ZIP containing: portfolios.csv, transactions.csv, baseline_lots.csv")

    st.download_button(
        "⬇️ Download CSV backup (ZIP)",
        data=build_backup_zip(),
        file_name=f"portfolio_backup_{date.today().isoformat()}.zip",
        mime="application/zip",
    )
