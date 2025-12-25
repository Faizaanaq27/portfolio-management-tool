import hmac
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from pathlib import Path
import streamlit as st

LOGO_PATH = Path(__file__).parent / "biglogo-white.png"

st.set_page_config(page_title="Multi-Portfolio Tracker (Cash + Snapshot)", layout="wide")

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

st.sidebar.image("biglogo-white.png", use_container_width=True)

# =========================
# 2) Storage (CSV)
# =========================
TXN_PATH = "transactions.csv"
PORTFOLIO_PATH = "portfolios.csv"
BASELINE_PATH = "baseline_lots.csv"

# Transactions ledger: buy/sell/dividend (dividend is cash-only)
TXN_COLS = ["txn_id", "portfolio", "date", "type", "ticker", "shares", "price", "amount"]

# Portfolio metadata supports two modes
PORTFOLIO_COLS = ["portfolio", "start_mode", "as_of_date", "starting_cash"]
VALID_MODES = ["ledger_complete", "snapshot_start"]

# Baseline lots (for snapshot portfolios)
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

    # Default portfolio
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

    # If as_of_date missing, default to today
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

    # buy/sell require ticker, shares, price
    is_trade = df["type"].isin(["buy", "sell"])
    trades = df[is_trade].dropna(subset=["ticker", "shares", "price"]).copy()
    trades = trades[(trades["shares"] > 0) & (trades["price"] >= 0)]

    # dividends require amount; ticker optional
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


# =========================
# 3) Lot engine + accounting
# =========================
def apply_sell_to_lots(lots: list[dict], sell_shares: float, method: str):
    """
    Mutates lots in-place (each lot has shares_open, buy_price, buy_date, lot_id, ticker).
    Returns list of realized match rows: (lot, shares_sold).
    """
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
    """
    Build open lots + realized matches using:
      - baseline lots as starting inventory
      - trades (buy/sell) as additional lots / reductions
    """
    open_cols = ["lot_id", "ticker", "buy_date", "buy_price", "shares_open"]
    real_cols = ["sale_id", "ticker", "buy_date", "buy_price", "sell_date", "sell_price", "shares_sold", "pnl"]

    lots_by_ticker = {}

    # Starting lots from baseline
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
    """
    Enforce:
      - In snapshot_start: no txns dated before as_of_date
      - Cash never negative (starting_cash + cashflows)
      - Cannot sell more shares than owned (baseline lots + buys - sells)
    """
    start_mode = portfolio_meta["start_mode"]
    as_of = pd.to_datetime(portfolio_meta["as_of_date"])
    starting_cash = float(portfolio_meta["starting_cash"])

    txns = portfolio_txns_all.copy()
    if not txns.empty:
        txns["date"] = pd.to_datetime(txns["date"])
        txns = txns.sort_values(["date", "txn_id"])

    # Snapshot boundary rule
    if start_mode == "snapshot_start" and not txns.empty:
        if (txns["date"] < as_of).any():
            return False, f"Snapshot portfolios cannot have transactions before as-of date ({as_of.date()})."

    # Cash constraint (baseline lots do NOT affect cash)
    cash = starting_cash
    for _, r in txns.iterrows():
        cash += cash_delta(r)
        if cash < -1e-9:
            return False, "Invalid: cash would go negative (margin not allowed)."

    # Share constraint: baseline + buys - sells
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
    """
    Returns metrics + tables for display.
    """
    as_of = pd.to_datetime(portfolio_meta["as_of_date"])
    start_mode = portfolio_meta["start_mode"]
    starting_cash = float(portfolio_meta["starting_cash"])

    df = txns.copy()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    # For snapshot portfolios, only consider txns on/after as_of
    if start_mode == "snapshot_start" and not df.empty:
        df = df[df["date"] >= as_of].copy()

    # Cash (baseline lots do not affect cash)
    cash_now = starting_cash + (df.apply(cash_delta, axis=1).sum() if not df.empty else 0.0)

    # Lots and realized from trades + baseline
    trades = df[df["type"].isin(["buy", "sell"])].copy() if not df.empty else pd.DataFrame(columns=TXN_COLS)
    open_lots, realized = build_lots_with_baseline(trades, baseline, method)

    # Live pricing + unrealized
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
    nav = cash_now + mv
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
# 4) Load data + reconcile
# =========================
portfolios_df = load_portfolios()
txns_all = load_txns()
baseline_all = load_baseline()

# Ensure portfolios referenced by txns exist
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
# 5) UI
# =========================

st.title("Brown Investment Group Portfolio")
st.caption("Welcome.")

st.sidebar.header("Lot settings")
match_method = st.sidebar.selectbox("Sell matching", ["FIFO", "LIFO"], index=0)

# ---------- Public view: tabs ----------
st.subheader("Public View")
tabs = st.tabs(portfolio_names)


def render_public_portfolio(pname: str):
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


for i, p in enumerate(portfolio_names):
    with tabs[i]:
        st.markdown(f"## {p}")
        render_public_portfolio(p)

# ---------- Admin section (REFINED UI) ----------
if is_admin:
    st.markdown("---")
    st.header("Admin")

    # One selector for the whole admin area
    active = st.selectbox("Active portfolio", portfolio_names, index=0, key="admin_active_portfolio")
    meta = get_portfolio_meta(portfolios_df, active)

    admin_tabs = st.tabs(["Portfolios", "Baseline lots", "Transactions", "Documentation"])

    # -----------------------
    # Portfolios tab
    # -----------------------
    with admin_tabs[0]:
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
        st.subheader(f"Edit settings: {active}")

        with st.form("edit_portfolio_form"):
            mode_u = st.selectbox("Start mode", VALID_MODES, index=VALID_MODES.index(meta["start_mode"]))
            asof_u = st.date_input("As-of date", value=meta["as_of_date"].date())
            cash_u = st.number_input(
                "Starting cash ($)", min_value=0.0, value=float(meta["starting_cash"]), step=100.0, format="%.2f"
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
    with admin_tabs[1]:
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
    # Transactions tab
    # -----------------------
    with admin_tabs[2]:
        st.subheader("Transactions")

        with st.form("add_txn_form", clear_on_submit=True):
            txn_type = st.selectbox("Type", ["buy", "sell", "dividend"], index=0)
            t_date = st.date_input("Date", value=date.today())

            if txn_type in ["buy", "sell"]:
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    t_ticker = st.text_input("Ticker", value="AAPL")
                with c2:
                    t_shares = st.number_input("Shares", min_value=0.0, value=1.000, step=0.001, format="%.3f")
                with c3:
                    t_price = st.number_input("Price", min_value=0.0, value=100.00, step=0.01, format="%.2f")
                t_amount = np.nan
            else:
                c1, c2 = st.columns([1, 1])
                with c1:
                    t_ticker = st.text_input("Ticker (optional)", value="")
                with c2:
                    t_amount = st.number_input("Dividend amount ($)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
                t_shares = np.nan
                t_price = np.nan

            add_txn = st.form_submit_button("Save transaction")

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
    # Documentation tab
    # -----------------------
    with admin_tabs[3]:
        st.subheader("Documentation")

        st.markdown("""
### Quick start

**If you have full history (ledger-complete):**
1. Go to **Portfolios** → set **Start mode** = `ledger_complete`
2. Set **Starting cash** (cash at the beginning of your ledger)
3. Enter **all buys/sells/dividends** in **Transactions**

**If you only have today's holdings (snapshot-start):**
1. Go to **Portfolios** → set **Start mode** = `snapshot_start`
2. Set **As-of date** = the snapshot boundary (usually today)
3. Set **Starting cash** = cash in the account on the as-of date
4. Go to **Baseline lots** → add each holding with:
   - ticker
   - shares
   - cost basis (price)
   - acquisition date (inception date)
5. Enter only **new** buys/sells/dividends after the as-of date in **Transactions**

---

### What each concept means

#### Start mode
- **`ledger_complete`**
  - You have the full transaction history (or at least everything you care about).
  - The app computes cash, lots, realized/unrealized P&L from the start of your ledger.

- **`snapshot_start`**
  - You *don’t* have full history.
  - You define a “starting state” at the **As-of date** using:
    - Starting cash
    - Baseline lots (current holdings)
  - The app tracks performance accurately **from the as-of date forward**.

#### As-of date
- The boundary date for snapshot portfolios.
- In `snapshot_start`, transactions **before** this date are blocked.

#### Starting cash
- The cash balance **at the as-of date**.
- Cash updates automatically from:
  - **Buy** → cash decreases
  - **Sell** → cash increases
  - **Dividend** → cash increases

#### Baseline lots
- Used only for `snapshot_start` portfolios.
- Represents positions that already existed on the as-of date.
- Baseline lots affect **shares owned** and **P&L**, but do **not** change cash (because the cash impact happened before the snapshot).

---

### Transactions

#### Buy
- Creates a new lot at (date, price, shares)
- Decreases cash by `shares * price`

#### Sell
- Closes lots using your chosen matching method (FIFO or LIFO)
- Increases cash by `shares * price`
- Realized P&L is computed lot-by-lot:
  - `(sell_price - buy_price) * shares_sold`
- The app blocks selling more shares than you own.

#### Dividend
- Immediately increases cash by the dividend amount
- Does not create/close lots (cash-only event)

---

### Sell matching (FIFO / LIFO)
- **FIFO**: sells the oldest lots first
- **LIFO**: sells the newest lots first
- This changes realized P&L timing (tax-style accounting), but not total long-run economics.

---

### Common errors

- **Cash would go negative**
  - Your buy exceeds available cash.
  - Fix by increasing Starting cash or reducing buy size.

- **Invalid SELL: not enough shares**
  - You tried to sell more shares than you own (including baseline lots).
  - Fix by adding the missing baseline lot / buy transaction, or reducing the sell.

- **Snapshot portfolios cannot have transactions before as-of date**
  - Move the transaction date forward, or switch portfolio to ledger_complete.
        """)
                    

st.caption(
    "Files used: portfolios.csv, baseline_lots.csv, transactions.csv. "
    "Snapshot portfolios track from as-of date forward; baseline lots represent holdings at the boundary."
)
