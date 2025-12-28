import hmac
from datetime import date
from pathlib import Path

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
BASELINE_COLS = ["lot_id", "portfolio", "ticker", "buy_date", "buy_price", "shares_open"]


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
    df = df[df["type"].isin(["buy", "sell", "dividend"])].copy()

    is_trade = df["type"].isin(["buy", "sell"])
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
    df = df[(df["shares_open"] > 0) & (df["buy_price"] >= 0)]
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
    # tz-safe -> tz-naive
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
    except Exception:
        # sometimes it's tz-localized; try localize(None)
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
# 3) Lot engine + accounting
# =========================
def apply_sell_to_lots(lots: list[dict], sell_shares: float, method: str):
    realized_rows = []
    remaining = sell_shares
    lot_iter = lots if method.upper() == "FIFO" else list(reversed(lots))

    i = 0
    while remaining > 1e-12 and i < len(lot_iter):
        lot = lot_iter[i]
        if lot["shares_open"] <= 1e-12:
            i += 1
            continue
        take = min(remaining, lot["shares_open"])
        lot["shares_open"] -= take
        remaining -= take
        realized_rows.append((lot, take))
    return realized_rows


def build_lots_with_baseline(trades: pd.DataFrame, baseline: pd.DataFrame, method: str):
    open_cols = ["lot_id", "ticker", "buy_date", "buy_price", "shares_open"]
    real_cols = ["sale_id", "ticker", "buy_date", "buy_price", "sell_date", "sell_price", "shares_sold", "pnl"]

    lots_by_ticker = {}

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
            open_lots.extend([x for x in lots if x["shares_open"] > 1e-12])
        return pd.DataFrame(open_lots, columns=open_cols), pd.DataFrame(columns=real_cols)

    trades = trades.copy().sort_values(["ticker", "date", "txn_id"])
    for _, r in trades.iterrows():
        typ = str(r["type"]).lower()
        tkr = str(r["ticker"]).upper().strip()

        if typ == "buy":
            lots_by_ticker.setdefault(tkr, []).append(
                {
                    "lot_id": str(r["txn_id"]),
                    "ticker": tkr,
                    "buy_date": pd.to_datetime(r["date"]).date(),
                    "buy_price": float(r["price"]),
                    "shares_open": float(r["shares"]),
                }
            )

        elif typ == "sell":
            sell_shares = float(r["shares"])
            sell_price = float(r["price"])
            sell_date = pd.to_datetime(r["date"]).date()
            sale_id = str(r["txn_id"])

            lots = lots_by_ticker.get(tkr, [])
            matches = apply_sell_to_lots(lots, sell_shares, method)

            for lot, shares_sold in matches:
                pnl = shares_sold * (sell_price - float(lot["buy_price"]))
                realized.append(
                    {
                        "sale_id": sale_id,
                        "ticker": tkr,
                        "buy_date": lot["buy_date"],
                        "buy_price": float(lot["buy_price"]),
                        "sell_date": sell_date,
                        "sell_price": sell_price,
                        "shares_sold": shares_sold,
                        "pnl": pnl,
                    }
                )

    open_lots = []
    for _, lots in lots_by_ticker.items():
        open_lots.extend([x for x in lots if x["shares_open"] > 1e-12])

    return pd.DataFrame(open_lots, columns=open_cols), pd.DataFrame(realized, columns=real_cols)


def cash_delta(row: pd.Series) -> float:
    typ = str(row["type"]).lower()
    if typ == "buy":
        return -float(row["shares"]) * float(row["price"])
    if typ == "sell":
        return +float(row["shares"]) * float(row["price"])
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

    trades = txns[txns["type"].isin(["buy", "sell"])].copy()
    if trades.empty:
        return True, ""

    inv = {}
    if not portfolio_baseline.empty:
        for _, r in portfolio_baseline.iterrows():
            t = str(r["ticker"]).upper().strip()
            inv[t] = inv.get(t, 0.0) + float(r["shares_open"])

    for _, r in trades.sort_values(["date", "txn_id"]).iterrows():
        t = str(r["ticker"]).upper().strip()
        sh = float(r["shares"])
        if r["type"] == "buy":
            inv[t] = inv.get(t, 0.0) + sh
        else:
            if inv.get(t, 0.0) + 1e-12 < sh:
                return False, f"Invalid SELL: not enough shares of {t} to sell {sh:.3f}."
            inv[t] -= sh

    return True, ""


def portfolio_snapshot(portfolio_meta: dict, txns: pd.DataFrame, baseline: pd.DataFrame, method: str):
    as_of = pd.to_datetime(portfolio_meta["as_of_date"])
    start_mode = portfolio_meta["start_mode"]
    starting_cash = float(portfolio_meta["starting_cash"])

    df = txns.copy()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    if start_mode == "snapshot_start" and not df.empty:
        df = df[df["date"] >= as_of].copy()

    cash_now = starting_cash + (df.apply(cash_delta, axis=1).sum() if not df.empty else 0.0)

    trades = df[df["type"].isin(["buy", "sell"])].copy() if not df.empty else pd.DataFrame(columns=TXN_COLS)
    open_lots, realized = build_lots_with_baseline(trades, baseline, method)

    if not open_lots.empty:
        tickers = sorted(open_lots["ticker"].unique().tolist())
        live = fetch_last_prices(tickers)

        lots_view = open_lots.copy()
        lots_view["last_price"] = lots_view["ticker"].map(live).astype(float)
        lots_view["market_value"] = lots_view["shares_open"] * lots_view["last_price"]
        lots_view["unrealized_pnl"] = lots_view["shares_open"] * (lots_view["last_price"] - lots_view["buy_price"])
        lots_view["unrealized_return_%"] = np.where(
            lots_view["buy_price"] > 0,
            ((lots_view["last_price"] / lots_view["buy_price"]) - 1.0) * 100.0,
            np.nan,
        )

        tmp = lots_view.copy()
        tmp["cost_dollars"] = tmp["shares_open"] * tmp["buy_price"]
        holdings = tmp.groupby("ticker", as_index=False).agg(
            shares=("shares_open", "sum"),
            cost_dollars=("cost_dollars", "sum"),
            market_value=("market_value", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        holdings["avg_cost"] = np.where(holdings["shares"] > 0, holdings["cost_dollars"] / holdings["shares"], np.nan)
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

    trades = txns[txns["type"].isin(["buy", "sell"])].copy() if not txns.empty else pd.DataFrame(columns=TXN_COLS)
    tickers = set(baseline_shares.keys())

    if not trades.empty:
        trades["ticker"] = trades["ticker"].astype(str).str.upper().str.strip()
        trades["signed_shares"] = np.where(trades["type"] == "buy", trades["shares"], -trades["shares"])
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
            rows.append(
                {"ticker": str(r["ticker"]).upper().strip(), "label": str(r["ticker"]).upper().strip(), "market_value": float(r["market_value"])}
            )

    rows.append({"ticker": "", "label": "CASH", "market_value": float(snap["cash"])})

    df = pd.DataFrame(rows)
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)

    total = float(df["market_value"].sum())
    df["weight"] = 0.0 if total <= 0 else (df["market_value"] / total)

    tickers = sorted([t for t in df["ticker"].unique().tolist() if t])
    meta = fetch_sector_industry(tickers) if tickers else pd.DataFrame(columns=["ticker", "sector", "industry"])

    df = df.merge(meta, on="ticker", how="left")
    df.loc[df["label"] == "CASH", "sector"] = "Cash"
    df.loc[df["label"] == "CASH", "industry"] = "Cash"
    df["sector"] = df["sector"].fillna("Unknown")
    df["industry"] = df["industry"].fillna("Unknown")

    sector_alloc = (
        df.groupby(["sector"], as_index=False)
        .agg(market_value=("market_value", "sum"))
        .sort_values("market_value", ascending=False)
    )
    sector_total = float(sector_alloc["market_value"].sum())
    sector_alloc["weight"] = np.where(sector_total > 0, sector_alloc["market_value"] / sector_total, 0.0)

    industry_alloc = (
        df.groupby(["sector", "industry"], as_index=False)
        .agg(market_value=("market_value", "sum"))
        .sort_values(["sector", "market_value"], ascending=[True, False])
    )
    ind_total = float(industry_alloc["market_value"].sum())
    industry_alloc["weight"] = np.where(ind_total > 0, industry_alloc["market_value"] / ind_total, 0.0)

    return sector_alloc, industry_alloc


def build_contribution_table(snap: dict) -> pd.DataFrame:
    unreal = pd.DataFrame(columns=["ticker", "unrealized_pnl"])
    if not snap["lots"].empty:
        unreal = snap["lots"].groupby("ticker", as_index=False).agg(unrealized_pnl=("unrealized_pnl", "sum"))

    realized = pd.DataFrame(columns=["ticker", "realized_pnl"])
    if not snap["realized"].empty:
        realized = snap["realized"].groupby("ticker", as_index=False).agg(realized_pnl=("pnl", "sum"))

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
    out["total_contribution"] = out["unrealized_pnl"] + out["realized_pnl"] + out["dividend_pnl"]
    out = out.sort_values("total_contribution", ascending=False).reset_index(drop=True)
    return out


def render_sector_pie(sector_alloc: pd.DataFrame, title: str):
    if sector_alloc.empty or sector_alloc["market_value"].sum() <= 0:
        st.info("No sector allocation available yet.")
        return
    fig, ax = plt.subplots()
    ax.pie(sector_alloc["market_value"], labels=sector_alloc["sector"], autopct="%1.1f%%", startangle=90)
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
    Daily shares outstanding by ticker from start boundary to end_date.
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

    trades = txns[txns["type"].isin(["buy", "sell"])].copy() if not txns.empty else pd.DataFrame(columns=TXN_COLS)
    tickers = set(baseline_shares.keys())
    if not trades.empty:
        trades["ticker"] = trades["ticker"].astype(str).str.upper().str.strip()
        trades["signed_shares"] = np.where(trades["type"] == "buy", trades["shares"], -trades["shares"])
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
    Uses yfinance dividend-per-share events and multiplies by shares held on that date.
    Groups cash dividends into quarter-ends.
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
            sh = float(shares_df.at[d, t])
            if sh <= 1e-12:
                continue
            events.append(
                {
                    "ticker": t,
                    "div_date": d,
                    "div_per_share": float(v),
                    "shares": sh,
                    "div_cash": float(v) * sh,
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
match_method = st.sidebar.selectbox("Sell matching", ["FIFO", "LIFO"], index=0)

freq_choice = st.sidebar.selectbox("Chart frequency", ["Weekly (Mon)", "Daily"], index=0)
chart_freq = "W-MON" if freq_choice.startswith("Weekly") else "D"

# ---------- Public view ----------
st.subheader("Public View (read-only)")
public_tabs = st.tabs(["Overview"] + portfolio_names)


def render_public_portfolio(pname: str, nav_series: pd.Series | None = None, spy_px: pd.Series | None = None):
    meta = get_portfolio_meta(portfolios_df, pname)
    p_txns = txns_all[txns_all["portfolio"] == pname].copy()
    p_base = baseline_all[baseline_all["portfolio"] == pname].copy()

    snap = portfolio_snapshot(meta, p_txns, p_base, match_method)

    if snap["start_mode"] == "snapshot_start":
        st.info(
            f"Snapshot portfolio — tracking boundary starts {snap['as_of'].date()}. "
            f"Baseline lots represent holdings as of that date."
        )
    else:
        st.caption("Ledger-complete portfolio — metrics reflect your ledger as entered.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Starting Cash", f"${snap['starting_cash']:,.2f}")
    c2.metric("Cash", f"${snap['cash']:,.2f}")
    c3.metric("Market Value", f"${snap['market_value']:,.2f}")
    c4.metric("NAV", f"${snap['nav']:,.2f}")

    d1, d2 = st.columns(2)
    d1.metric("Unrealized P&L", f"${snap['unrealized_pnl']:,.2f}")
    d2.metric("Realized P&L", f"${snap['realized_pnl']:,.2f}")

    # Dividend accrual (computed dividends, not necessarily entered as txns)
    divpack = compute_dividend_accrual_quarterly(pname, meta, txns_all, baseline_all, pd.to_datetime(date.today()))
    st.metric("Dividends (estimated, quarterly accrual)", f"${float(divpack['total']):,.2f}")

    st.divider()
    st.markdown("## Tier 1 Analytics")

    sector_alloc, industry_alloc = build_allocation_tables(snap)
    a1, a2 = st.columns([1, 1])
    with a1:
        st.markdown("### Sector allocation (incl. cash)")
        render_sector_pie(sector_alloc, f"{pname} — Sector Allocation")
        st.dataframe(
            sector_alloc.assign(weight_pct=(sector_alloc["weight"] * 100).round(2)).drop(columns=["weight"]),
            use_container_width=True,
        )
    with a2:
        st.markdown("### Sector → Industry breakdown (Yahoo Finance)")
        st.dataframe(
            industry_alloc.assign(weight_pct=(industry_alloc["weight"] * 100).round(2)).drop(columns=["weight"]),
            use_container_width=True,
        )

    st.markdown("### Contribution to return (P&L contribution by ticker)")
    contrib = build_contribution_table(snap)
    if contrib.empty:
        st.info("No contribution data yet (need holdings and/or sells/dividends).")
    else:
        st.dataframe(contrib, use_container_width=True)
        chart_df = contrib.set_index("ticker")[["total_contribution"]]
        st.bar_chart(chart_df)

    st.markdown("### Dividend tracker (estimated)")
    if divpack["quarterly"] is None or divpack["quarterly"].empty:
        st.info("No dividend events found for held tickers in this period.")
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
        st.markdown("### Holdings")
        st.dataframe(snap["holdings"], use_container_width=True)

    if not snap["lots"].empty:
        st.markdown("### Open lots (baseline + buys)")
        st.dataframe(snap["lots"].sort_values(["ticker", "buy_date"]), use_container_width=True)

    if not snap["realized"].empty:
        rv = snap["realized"].copy()
        rv["realized_return_%"] = np.where(
            rv["buy_price"] > 0,
            ((rv["sell_price"] / rv["buy_price"]) - 1.0) * 100.0,
            np.nan,
        )
        st.markdown("### Realized matches (per lot)")
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

    end_date = pd.to_datetime(date.today())
    nav_series_map = {}
    cash_now_by_port = {}
    pnl_now_by_port = {}
    div_total_by_port = {}

    for p in portfolio_names:
        meta = get_portfolio_meta(portfolios_df, p)
        p_txns = txns_all[txns_all["portfolio"] == p].copy()
        p_base = baseline_all[baseline_all["portfolio"] == p].copy()

        snap = portfolio_snapshot(meta, p_txns, p_base, match_method)
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

        render_public_portfolio(p, nav_series=nav, spy_px=spy_aligned)

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
        new_asof = st.date_input("As-of date", value=date.today())
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

    active = st.selectbox("Active portfolio", portfolio_names, index=0, key="admin_active_portfolio")
    meta = get_portfolio_meta(portfolios_df, active)

    admin_tabs = st.tabs(["Transactions", "Portfolio Settings", "Baseline lots", "Documentation"])

    # -----------------------
    # Transactions tab
    # -----------------------
    with admin_tabs[0]:
        st.subheader("Transactions")

        with st.form("add_txn_form", clear_on_submit=False):
            txn_type = st.selectbox("Type", ["buy", "sell", "dividend"], index=0)
            t_date = st.date_input("Date", value=date.today())

            # NOTE: two submit buttons in the same form
            refresh_pressed = False
            add_txn = False

            if txn_type in ["buy", "sell"]:
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    t_ticker = st.text_input("Ticker", value="AAPL", key=f"txn_ticker__{active}")
                with c2:
                    t_shares = st.number_input(
                        "Shares", min_value=0.0, value=1.000, step=0.001, format="%.3f", key=f"txn_shares__{active}"
                    )

                t_amount = np.nan

                tkr_clean = _clean_str(t_ticker).upper()
                chosen_date = t_date

                rec_key = f"use_recommended_price__{active}"
                if rec_key not in st.session_state:
                    st.session_state[rec_key] = True
                st.checkbox("Use recommended close price", key=rec_key)

                rec = fetch_close_on_or_before(tkr_clean, chosen_date) if tkr_clean else None
                price_key = f"txn_price__{active}"

                # seed if not present
                if price_key not in st.session_state:
                    st.session_state[price_key] = float(rec) if (st.session_state[rec_key] and rec is not None) else 100.00
                else:
                    # if toggle is ON, keep price synced to rec when ticker/date changes
                    # (simple approach: overwrite whenever rec exists and toggle on)
                    if st.session_state[rec_key] and rec is not None:
                        st.session_state[price_key] = float(rec)

                with c3:
                    t_price = st.number_input(
                        "Price",
                        min_value=0.0,
                        value=float(st.session_state[price_key]),
                        step=0.01,
                        format="%.2f",
                        key=price_key,
                        help="Auto-fills with close on/before the transaction date (editable).",
                    )
                    refresh_pressed = st.form_submit_button("🔄 Refresh price")

                if rec is not None and tkr_clean:
                    st.caption(f"Recommended close for {tkr_clean} on/before {chosen_date.isoformat()}: **${rec:,.2f}**")

            else:
                c1, c2 = st.columns([1, 1])
                with c1:
                    t_ticker = st.text_input("Ticker (optional)", value="")
                with c2:
                    t_amount = st.number_input("Dividend amount ($)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
                t_shares = np.nan
                t_price = np.nan

            add_txn = st.form_submit_button("Save transaction")

        # handle refresh (outside the form block)
        if refresh_pressed:
            tkr_clean = _clean_str(st.session_state.get(f"txn_ticker__{active}", "")).upper()
            chosen_date = t_date
            price_key = f"txn_price__{active}"
            if not tkr_clean:
                st.warning("Enter a ticker first.")
            else:
                rec2 = fetch_close_on_or_before(tkr_clean, chosen_date)
                if rec2 is None:
                    st.error("Could not fetch a close price for that ticker/date.")
                else:
                    st.session_state[price_key] = float(rec2)
                    st.success(f"Updated price to ${rec2:,.2f}")
                    st.rerun()

        if add_txn:
            row = {
                "txn_id": str(pd.Timestamp.utcnow().value),
                "portfolio": active,
                "date": pd.to_datetime(t_date),
                "type": txn_type,
                "ticker": _clean_str(t_ticker).upper(),
                "shares": float(round(t_shares, 3)) if pd.notna(t_shares) else np.nan,
                "price": float(t_price) if pd.notna(t_price) else np.nan,
                "amount": float(t_amount) if pd.notna(t_amount) else np.nan,
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
        st.subheader("Delete transaction")

        p_txns = txns_all[txns_all["portfolio"] == active].copy()
        if p_txns.empty:
            st.info("No transactions.")
        else:
            disp = p_txns.copy()
            disp["date_str"] = pd.to_datetime(disp["date"]).dt.date.astype(str)
            disp["desc"] = np.where(
                disp["type"].isin(["buy", "sell"]),
                disp["date_str"]
                + " | "
                + disp["type"]
                + " | "
                + disp["ticker"]
                + " | "
                + disp["shares"].map(lambda x: f"{x:.3f}")
                + " @ "
                + disp["price"].map(lambda x: f"{x:.2f}"),
                disp["date_str"]
                + " | dividend | "
                + disp["ticker"].fillna("").astype(str)
                + " | $"
                + disp["amount"].map(lambda x: f"{x:.2f}"),
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
            asof_u = st.date_input("As-of date", value=meta["as_of_date"].date())
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
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                with c1:
                    bl_ticker = st.text_input("Ticker", value="AAPL")
                with c2:
                    bl_shares = st.number_input("Shares", min_value=0.0, value=1.000, step=0.001, format="%.3f")
                with c3:
                    bl_price = st.number_input("Cost basis (price)", min_value=0.0, value=100.00, step=0.01, format="%.2f")
                with c4:
                    bl_date = st.date_input("Acquisition date", value=date.today())

                add_bl = st.form_submit_button("Add baseline lot")

            if add_bl:
                row = {
                    "lot_id": str(pd.Timestamp.utcnow().value),
                    "portfolio": active,
                    "ticker": _clean_str(bl_ticker).upper(),
                    "buy_date": pd.to_datetime(bl_date),
                    "buy_price": float(bl_price),
                    "shares_open": float(round(bl_shares, 3)),
                }
                baseline_all = pd.concat([baseline_all, pd.DataFrame([row])], ignore_index=True)
                save_baseline(baseline_all)
                st.success("Baseline lot added.")
                st.rerun()

            st.divider()
            st.write("Current baseline lots")
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
3. Enter **all buys/sells/dividends** in **Transactions**

**If you only have today's holdings (snapshot-start):**
1. Go to **Portfolio Settings** → set **Start mode** = `snapshot_start`
2. Set **As-of date** = the snapshot boundary (usually today)
3. Set **Starting cash** = cash in the account on the as-of date
4. Go to **Baseline lots** → add each holding with:
   - ticker
   - shares
   - cost basis (price)
   - acquisition date (inception date)
5. Enter only **new** buys/sells/dividends after the as-of date in **Transactions**

---

### Charting notes
- Charts start at the portfolio **As-of date** (no line before the start date).
- Weekly mode samples on **Mondays**, but always includes the start date.

---

### Dividend tracker notes
- Dividends are **estimated** using Yahoo dividend-per-share events × shares held on that date.
- They are **bucketed into quarter-ends** for a simple “quarterly accrual” view.

---

### Common errors

- **Cash would go negative**
  - Buy exceeds available cash.
  - Increase Starting cash or reduce buy size.

- **Invalid SELL: not enough shares**
  - You tried to sell more than you own.
  - Add missing baseline/buy entries or reduce the sell.

- **Snapshot portfolios cannot have transactions before as-of date**
  - Move the transaction date forward, or switch to ledger_complete.
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
