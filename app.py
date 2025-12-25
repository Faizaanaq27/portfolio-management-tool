import hmac
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Multi-Portfolio Lots Tracker (with Cash)", layout="wide")

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

# Transactions are a ledger. "type" can be: buy, sell, dividend
TXN_COLS = ["txn_id", "portfolio", "date", "type", "ticker", "shares", "price", "amount"]

# Each portfolio has starting cash
PORTFOLIO_COLS = ["portfolio", "starting_cash"]


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
    df["starting_cash"] = pd.to_numeric(df["starting_cash"], errors="coerce").fillna(0.0)

    df = df[df["portfolio"].notna() & (df["portfolio"] != "")]
    if "Main" not in set(df["portfolio"].tolist()):
        df = pd.concat([pd.DataFrame([{"portfolio": "Main", "starting_cash": 0.0}]), df], ignore_index=True)

    df = df.drop_duplicates(subset=["portfolio"]).sort_values("portfolio").reset_index(drop=True)
    return df


def save_portfolios(df: pd.DataFrame) -> None:
    out = df.copy()
    out["portfolio"] = out["portfolio"].astype(str).str.strip()
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

    # Normalize required fields per type
    # buy/sell require ticker, shares, price
    mask_trade = df["type"].isin(["buy", "sell"])
    df_trades = df[mask_trade].copy()
    df_trades = df_trades.dropna(subset=["ticker", "shares", "price"])
    df_trades = df_trades[(df_trades["shares"] > 0) & (df_trades["price"] >= 0)]

    # dividend requires amount; ticker optional
    df_div = df[~mask_trade].copy()
    df_div = df_div.dropna(subset=["amount"])
    df_div = df_div[df_div["amount"] >= 0]
    # allow blank ticker for dividends (portfolio-level)
    df_div["ticker"] = df_div["ticker"].replace({"NAN": "", "NONE": ""}).fillna("")
    df_div.loc[df_div["ticker"] == "NAN", "ticker"] = ""

    df = pd.concat([df_trades, df_div], ignore_index=True)
    return df.sort_values(["portfolio", "date", "txn_id"]).reset_index(drop=True)


def save_txns(df: pd.DataFrame) -> None:
    out = df.copy()
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out.to_csv(TXN_PATH, index=False)


@st.cache_data(ttl=600)
def fetch_last_prices(tickers: list[str]) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)

    data = yf.download(
        tickers,
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[-1]
    else:
        close = pd.Series({tickers[0]: data["Close"].iloc[-1]})

    close.index = [str(x).upper() for x in close.index]
    return close.astype(float)


# =========================
# 3) Core accounting logic
# =========================
def build_lots(trades: pd.DataFrame, method: str = "FIFO"):
    """
    Lot-based accounting for BUY/SELL only.
    Returns open lots + realized matches.
    """
    open_cols = ["lot_id", "ticker", "buy_date", "buy_price", "shares_open"]
    real_cols = ["sale_id", "ticker", "buy_date", "buy_price", "sell_date", "sell_price", "shares_sold", "pnl"]

    if trades.empty:
        return pd.DataFrame(columns=open_cols), pd.DataFrame(columns=real_cols)

    trades = trades.copy()
    trades["ticker"] = trades["ticker"].str.upper().str.strip()
    trades["type"] = trades["type"].str.lower().str.strip()
    trades = trades.sort_values(["ticker", "date", "txn_id"])

    open_lots = []
    realized = []

    for tkr, g in trades.groupby("ticker"):
        lots = []
        for _, r in g.iterrows():
            if r["type"] == "buy":
                lots.append({
                    "lot_id": r["txn_id"],
                    "ticker": tkr,
                    "buy_date": r["date"].date(),
                    "buy_price": float(r["price"]),
                    "shares_open": float(r["shares"]),
                })
            elif r["type"] == "sell":
                sell_shares = float(r["shares"])
                sell_price = float(r["price"])
                sell_date = r["date"].date()
                sale_id = r["txn_id"]

                lot_iter = lots if method.upper() == "FIFO" else list(reversed(lots))

                i = 0
                while sell_shares > 1e-12 and i < len(lot_iter):
                    lotref = lot_iter[i]
                    if lotref["shares_open"] <= 1e-12:
                        i += 1
                        continue

                    take = min(sell_shares, lotref["shares_open"])
                    pnl = take * (sell_price - lotref["buy_price"])

                    realized.append({
                        "sale_id": sale_id,
                        "ticker": tkr,
                        "buy_date": lotref["buy_date"],
                        "buy_price": lotref["buy_price"],
                        "sell_date": sell_date,
                        "sell_price": sell_price,
                        "shares_sold": take,
                        "pnl": pnl,
                    })

                    lotref["shares_open"] -= take
                    sell_shares -= take

        open_lots.extend([x for x in lots if x["shares_open"] > 1e-12])

    return pd.DataFrame(open_lots, columns=open_cols), pd.DataFrame(realized, columns=real_cols)


def compute_cash_delta(row: pd.Series) -> float:
    t = str(row["type"]).lower()
    if t == "buy":
        return -float(row["shares"]) * float(row["price"])
    if t == "sell":
        return +float(row["shares"]) * float(row["price"])
    if t == "dividend":
        return +float(row["amount"])
    return 0.0


def validate_portfolio_ledger(portfolio_txns: pd.DataFrame, starting_cash: float, method: str) -> tuple[bool, str]:
    """
    Validate:
      1) Cash never goes negative (disallow margin)
      2) You never sell more shares than owned (per ticker, using lots)

    Returns (ok, message).
    """
    if portfolio_txns.empty:
        if starting_cash < -1e-9:
            return False, "Starting cash cannot be negative."
        return True, ""

    df = portfolio_txns.copy().sort_values(["date", "txn_id"])
    cash = float(starting_cash)

    # Track lots per ticker for validation (same matching method)
    lots_by_ticker = {}

    for _, r in df.iterrows():
        typ = str(r["type"]).lower()

        if typ in ["buy", "sell"]:
            tkr = str(r["ticker"]).upper().strip()
            sh = float(r["shares"])
            px = float(r["price"])

            if typ == "buy":
                cash -= sh * px
                lots_by_ticker.setdefault(tkr, []).append({"shares": sh, "price": px})

            else:  # sell
                cash += sh * px
                if tkr not in lots_by_ticker or sum(l["shares"] for l in lots_by_ticker[tkr]) + 1e-12 < sh:
                    return False, f"Invalid SELL: not enough shares of {tkr} to sell {sh:.3f}."

                # Consume shares from lots (FIFO/LIFO)
                lots = lots_by_ticker[tkr]
                lot_iter = lots if method.upper() == "FIFO" else list(reversed(lots))

                remaining = sh
                i = 0
                while remaining > 1e-12 and i < len(lot_iter):
                    lot = lot_iter[i]
                    if lot["shares"] <= 1e-12:
                        i += 1
                        continue
                    take = min(remaining, lot["shares"])
                    lot["shares"] -= take
                    remaining -= take

        elif typ == "dividend":
            amt = float(r["amount"])
            cash += amt

        if cash < -1e-9:
            return False, "Invalid BUY: would make cash go negative (margin not allowed)."

    return True, ""


def portfolio_snapshot(portfolio_txns: pd.DataFrame, starting_cash: float, match_method: str):
    """
    Returns:
      cash_now, open_lots_view, realized_view, holdings_summary
    """
    if portfolio_txns.empty:
        return float(starting_cash), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = portfolio_txns.copy().sort_values(["date", "txn_id"])
    cash_now = float(starting_cash) + float(df.apply(compute_cash_delta, axis=1).sum())

    trades = df[df["type"].isin(["buy", "sell"])].copy()
    open_lots, realized = build_lots(trades, method=match_method)

    if not open_lots.empty:
        tickers = sorted(open_lots["ticker"].unique().tolist())
        live = fetch_last_prices(tickers)

        open_lots_view = open_lots.copy()
        open_lots_view["last_price"] = open_lots_view["ticker"].map(live).astype(float)
        open_lots_view["market_value"] = open_lots_view["shares_open"] * open_lots_view["last_price"]
        open_lots_view["unrealized_pnl"] = open_lots_view["shares_open"] * (
            open_lots_view["last_price"] - open_lots_view["buy_price"]
        )
        open_lots_view["unrealized_return_%"] = np.where(
            open_lots_view["buy_price"] > 0,
            ((open_lots_view["last_price"] / open_lots_view["buy_price"]) - 1.0) * 100.0,
            np.nan,
        )

        holdings = (
            open_lots_view.groupby("ticker", as_index=False)
            .agg(shares=("shares_open", "sum"),
                 cost_basis=("buy_price", lambda s: np.nan))  # placeholder
        )

        # Weighted avg cost for holdings
        tmp = open_lots_view.copy()
        tmp["cost_dollars"] = tmp["shares_open"] * tmp["buy_price"]
        hold = tmp.groupby("ticker", as_index=False).agg(
            shares=("shares_open", "sum"),
            cost_dollars=("cost_dollars", "sum"),
            market_value=("market_value", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        hold["avg_cost"] = np.where(hold["shares"] > 0, hold["cost_dollars"] / hold["shares"], np.nan)

        return cash_now, open_lots_view, realized, hold.sort_values("market_value", ascending=False)

    return cash_now, open_lots, realized, pd.DataFrame()


# =========================
# 4) App UI
# =========================
st.title("Manual Portfolio Tracker (Lot-Based, Multi-Portfolio, Cash)")

portfolios_df = load_portfolios()
txns_all = load_txns()

# Ensure any portfolios found in transactions exist in portfolios.csv
if not txns_all.empty:
    existing = set(portfolios_df["portfolio"].tolist())
    found = set(txns_all["portfolio"].astype(str).str.strip().tolist())
    missing = sorted([p for p in found if p and p not in existing])
    if missing:
        portfolios_df = pd.concat(
            [portfolios_df, pd.DataFrame([{"portfolio": p, "starting_cash": 0.0} for p in missing])],
            ignore_index=True,
        )
        save_portfolios(portfolios_df)

portfolio_names = portfolios_df["portfolio"].tolist()

st.sidebar.header("Lot settings")
match_method = st.sidebar.selectbox("Sell matching", ["FIFO", "LIFO"], index=0)

# -------- Public tabs --------
st.subheader("Public View (read-only)")
tabs = st.tabs(portfolio_names)

def render_portfolio_view(portfolio_name: str):
    p_start_cash = float(
        portfolios_df.loc[portfolios_df["portfolio"] == portfolio_name, "starting_cash"].iloc[0]
        if portfolio_name in set(portfolios_df["portfolio"])
        else 0.0
    )
    p_txns = txns_all[txns_all["portfolio"] == portfolio_name].copy()

    cash_now, open_lots_view, realized, holdings = portfolio_snapshot(p_txns, p_start_cash, match_method)

    # Portfolio metrics
    mv = float(open_lots_view["market_value"].sum()) if not open_lots_view.empty else 0.0
    nav = cash_now + mv
    realized_pnl = float(realized["pnl"].sum()) if not realized.empty else 0.0
    unrealized_pnl = float(open_lots_view["unrealized_pnl"].sum()) if not open_lots_view.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Starting Cash", f"${p_start_cash:,.2f}")
    k2.metric("Cash", f"${cash_now:,.2f}")
    k3.metric("Market Value", f"${mv:,.2f}")
    k4.metric("NAV", f"${nav:,.2f}")

    a1, a2 = st.columns(2)
    a1.metric("Unrealized P&L", f"${unrealized_pnl:,.2f}")
    a2.metric("Realized P&L", f"${realized_pnl:,.2f}")

    if p_txns.empty:
        st.info("No transactions yet in this portfolio.")
        return

    if not holdings.empty:
        st.markdown("### Holdings")
        st.dataframe(holdings, use_container_width=True)

    if not open_lots_view.empty:
        st.markdown("### Open lots (unrealized)")
        st.dataframe(open_lots_view.sort_values(["ticker", "buy_date"]), use_container_width=True)

    if not realized.empty:
        rv = realized.copy()
        rv["realized_return_%"] = np.where(
            rv["buy_price"] > 0,
            ((rv["sell_price"] / rv["buy_price"]) - 1.0) * 100.0,
            np.nan,
        )
        st.markdown("### Realized matches (per lot)")
        st.dataframe(rv.sort_values(["sell_date", "ticker"], ascending=False), use_container_width=True)

    st.markdown("### Transactions (read-only)")
    st.dataframe(p_txns.sort_values("date", ascending=False), use_container_width=True)

for i, p in enumerate(portfolio_names):
    with tabs[i]:
        st.markdown(f"## {p}")
        render_portfolio_view(p)

# -------- Admin section --------
if is_admin:
    st.markdown("---")
    st.subheader("Admin (edit enabled)")

    colA, colB = st.columns([1, 2])

    # Create portfolio + starting cash
    with colA:
        st.markdown("#### Create a new portfolio")
        new_name = st.text_input("Portfolio name", value="", placeholder="e.g., Long Only, Trading, IRA")
        new_cash = st.number_input("Starting cash ($)", min_value=0.0, value=0.0, step=100.0, format="%.2f")

        if st.button("Add portfolio", type="primary"):
            name = (new_name or "").strip()
            if not name:
                st.error("Enter a portfolio name.")
            elif name in set(portfolio_names):
                st.warning("That portfolio already exists.")
            else:
                portfolios_df = pd.concat(
                    [portfolios_df, pd.DataFrame([{"portfolio": name, "starting_cash": float(new_cash)}])],
                    ignore_index=True,
                )
                save_portfolios(portfolios_df)
                st.success("Portfolio added.")
                st.rerun()

        st.markdown("---")
        st.markdown("#### Update starting cash")
        p_sel = st.selectbox("Portfolio", portfolio_names, index=0, key="cash_portfolio_sel")
        current_cash = float(portfolios_df.loc[portfolios_df["portfolio"] == p_sel, "starting_cash"].iloc[0])
        updated_cash = st.number_input(
            "New starting cash ($)",
            min_value=0.0,
            value=float(current_cash),
            step=100.0,
            format="%.2f",
            key="cash_update_val",
        )
        if st.button("Save starting cash"):
            portfolios_df.loc[portfolios_df["portfolio"] == p_sel, "starting_cash"] = float(updated_cash)
            save_portfolios(portfolios_df)
            st.success("Updated starting cash.")
            st.rerun()

    # Add/Delete transactions (with cash validation)
    with colB:
        st.markdown("#### Add transaction")
        active_portfolio = st.selectbox("Active portfolio", portfolio_names, index=0, key="active_portfolio")

        txn_type = st.selectbox("Transaction type", ["buy", "sell", "dividend"], index=0)

        t_date = st.date_input("Date", value=date.today(), key="t_date")

        if txn_type in ["buy", "sell"]:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                t_ticker = st.text_input("Ticker", value="AAPL", key="t_ticker")
            with c2:
                t_shares = st.number_input(
                    "Shares (3 decimals)",
                    min_value=0.0,
                    value=1.000,
                    step=0.001,
                    format="%.3f",
                    key="t_shares",
                )
            with c3:
                t_price = st.number_input(
                    "Price",
                    min_value=0.0,
                    value=100.00,
                    step=0.01,
                    format="%.2f",
                    key="t_price",
                )
            amount = np.nan

        else:  # dividend
            c1, c2 = st.columns([1, 1])
            with c1:
                t_ticker = st.text_input("Ticker (optional)", value="", key="d_ticker")
            with c2:
                amount = st.number_input("Dividend amount ($)", min_value=0.0, value=0.0, step=1.0, format="%.2f", key="d_amt")
            t_shares = np.nan
            t_price = np.nan

        # Pull starting cash for portfolio
        p_start_cash = float(portfolios_df.loc[portfolios_df["portfolio"] == active_portfolio, "starting_cash"].iloc[0])

        if st.button("Save transaction", type="primary", key="save_txn"):
            row = {
                "txn_id": str(pd.Timestamp.utcnow().value),
                "portfolio": active_portfolio,
                "date": pd.to_datetime(t_date),
                "type": txn_type,
                "ticker": (t_ticker or "").strip().upper(),
                "shares": float(round(t_shares, 3)) if pd.notna(t_shares) else np.nan,
                "price": float(t_price) if pd.notna(t_price) else np.nan,
                "amount": float(amount) if pd.notna(amount) else np.nan,
            }

            # Insert and validate full ledger
            candidate = pd.concat([txns_all, pd.DataFrame([row])], ignore_index=True)

            p_txns = candidate[candidate["portfolio"] == active_portfolio].copy()
            ok, msg = validate_portfolio_ledger(p_txns, p_start_cash, match_method)

            if not ok:
                st.error(msg)
            else:
                txns_all = candidate
                save_txns(txns_all)
                st.success("Saved.")
                st.rerun()

        st.markdown("---")
        st.markdown("#### Delete transaction")
        txns_active = txns_all[txns_all["portfolio"] == active_portfolio].copy()

        if txns_active.empty:
            st.info("No transactions in this portfolio.")
        else:
            disp = txns_active.copy()
            disp["date_str"] = pd.to_datetime(disp["date"]).dt.date.astype(str)
            disp["desc"] = np.where(
                disp["type"].isin(["buy","sell"]),
                disp["date_str"] + " | " + disp["type"] + " | " + disp["ticker"] + " | "
                + disp["shares"].map(lambda x: f"{x:.3f}") + " @ " + disp["price"].map(lambda x: f"{x:.2f}"),
                disp["date_str"] + " | dividend | " + disp["ticker"].fillna("").astype(str) + " | $" + disp["amount"].map(lambda x: f"{x:.2f}")
            )
            disp["display"] = disp["desc"] + " | id=" + disp["txn_id"].astype(str)

            choice = st.selectbox("Select transaction", disp["display"].tolist(), key="delete_choice")
            chosen_id = choice.split("id=")[-1].strip()

            if st.button("Delete selected", key="delete_btn"):
                candidate = txns_all[txns_all["txn_id"].astype(str) != chosen_id].copy()

                # Validate ledger after delete as well (should always pass, but keeps invariants)
                p_start_cash = float(portfolios_df.loc[portfolios_df["portfolio"] == active_portfolio, "starting_cash"].iloc[0])
                p_txns = candidate[candidate["portfolio"] == active_portfolio].copy()
                ok, msg = validate_portfolio_ledger(p_txns, p_start_cash, match_method)

                if not ok:
                    st.error(f"Delete rejected: {msg}")
                else:
                    txns_all = candidate
                    save_txns(txns_all)
                    st.warning("Deleted.")
                    st.rerun()

st.caption("Note: This uses CSV files (transactions.csv + portfolios.csv). For production persistence, swap to SQLite/Postgres.")
