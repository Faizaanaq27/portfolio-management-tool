import hmac
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Multi-Portfolio Lots Tracker", layout="wide")

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
# 2) Storage
# =========================
TXN_PATH = "transactions.csv"
PORTFOLIO_PATH = "portfolios.csv"

TXN_COLS = ["txn_id", "portfolio", "ticker", "date", "side", "shares", "price"]
PORTFOLIO_COLS = ["portfolio"]


def load_portfolios() -> pd.DataFrame:
    """Persist portfolio names even if they have no transactions yet."""
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

    # ensure default
    if "Main" not in set(df["portfolio"].tolist()):
        df = pd.concat([pd.DataFrame([{"portfolio": "Main"}]), df], ignore_index=True)

    df = df.drop_duplicates().sort_values("portfolio").reset_index(drop=True)
    return df


def save_portfolios(df: pd.DataFrame) -> None:
    out = df.copy()
    out["portfolio"] = out["portfolio"].astype(str).str.strip()
    out = out[out["portfolio"].notna() & (out["portfolio"] != "")]
    out = out.drop_duplicates().sort_values("portfolio")
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
    df.loc[df["portfolio"].isna() | (df["portfolio"] == "") , "portfolio"] = "Main"

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["side"] = df["side"].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["txn_id"] = df["txn_id"].astype(str)

    df = df.dropna(subset=["portfolio", "ticker", "side", "date", "shares", "price"])
    df = df[df["side"].isin(["buy", "sell"])]
    df = df[df["shares"] > 0]
    df = df[df["price"] >= 0]

    return df.sort_values(["portfolio", "ticker", "date", "txn_id"])


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


def build_lots(txns: pd.DataFrame, method: str = "FIFO"):
    """Lot-based accounting for a single portfolio's transactions."""
    open_cols = ["lot_id", "ticker", "buy_date", "buy_price", "shares_open"]
    real_cols = [
        "sale_id",
        "ticker",
        "buy_date",
        "buy_price",
        "sell_date",
        "sell_price",
        "shares_sold",
        "pnl",
    ]

    if txns.empty:
        return pd.DataFrame(columns=open_cols), pd.DataFrame(columns=real_cols)

    txns = txns.copy()
    txns["ticker"] = txns["ticker"].str.upper().str.strip()
    txns["side"] = txns["side"].str.lower().str.strip()
    txns = txns.sort_values(["ticker", "date", "txn_id"])

    open_lots = []
    realized = []

    for tkr, g in txns.groupby("ticker"):
        lots = []

        for _, r in g.iterrows():
            if r["side"] == "buy":
                lots.append(
                    {
                        "lot_id": r["txn_id"],
                        "ticker": tkr,
                        "buy_date": r["date"].date(),
                        "buy_price": float(r["price"]),
                        "shares_open": float(r["shares"]),
                    }
                )
            else:  # sell
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

                    realized.append(
                        {
                            "sale_id": sale_id,
                            "ticker": tkr,
                            "buy_date": lotref["buy_date"],
                            "buy_price": lotref["buy_price"],
                            "sell_date": sell_date,
                            "sell_price": sell_price,
                            "shares_sold": take,
                            "pnl": pnl,
                        }
                    )

                    lotref["shares_open"] -= take
                    sell_shares -= take

        open_lots.extend([x for x in lots if x["shares_open"] > 1e-12])

    return pd.DataFrame(open_lots, columns=open_cols), pd.DataFrame(realized, columns=real_cols)


# =========================
# 3) App UI
# =========================
st.title("Manual Portfolio Tracker (Lot-Based, Multi-Portfolio)")

portfolios_df = load_portfolios()
txns_all = load_txns()

# Ensure any portfolios found in transactions also appear in portfolios.csv
if not txns_all.empty:
    existing = set(portfolios_df["portfolio"].tolist())
    found = set(txns_all["portfolio"].astype(str).str.strip().tolist())
    missing = sorted([p for p in found if p and p not in existing])
    if missing:
        portfolios_df = pd.concat(
            [portfolios_df, pd.DataFrame([{"portfolio": p} for p in missing])],
            ignore_index=True,
        )
        save_portfolios(portfolios_df)

portfolio_names = portfolios_df["portfolio"].tolist()

st.sidebar.header("Lot settings")
match_method = st.sidebar.selectbox("Sell matching", ["FIFO", "LIFO"], index=0)

# --- Public: show each portfolio as a tab ---
st.subheader("Public View (read-only)")
tabs = st.tabs(portfolio_names)

def render_portfolio_view(portfolio_name: str):
    txns = txns_all[txns_all["portfolio"] == portfolio_name].copy()
    open_lots, realized = build_lots(txns, method=match_method)

    if txns.empty:
        st.info("No transactions yet in this portfolio.")
        return

    # Open lots
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

        m1, m2, m3 = st.columns(3)
        m1.metric("Open lots", f"{len(lots_view)}")
        m2.metric("Unrealized P&L", f"${lots_view['unrealized_pnl'].sum():,.2f}")
        m3.metric("Open Market Value", f"${lots_view['market_value'].sum():,.2f}")

        st.write("**Open lots (unrealized)**")
        st.dataframe(lots_view.sort_values(["ticker", "buy_date"]), use_container_width=True)

    # Realized
    if not realized.empty:
        realized_view = realized.copy()
        realized_view["realized_return_%"] = np.where(
            realized_view["buy_price"] > 0,
            ((realized_view["sell_price"] / realized_view["buy_price"]) - 1.0) * 100.0,
            np.nan,
        )
        st.write("**Realized matches (each SELL matched to specific BUY lots)**")
        st.dataframe(realized_view.sort_values(["sell_date", "ticker"], ascending=False), use_container_width=True)
        st.metric("Realized P&L", f"${realized_view['pnl'].sum():,.2f}")

    st.markdown("---")
    st.write("**Transactions (read-only)**")
    st.dataframe(txns.sort_values("date", ascending=False), use_container_width=True)

for i, p in enumerate(portfolio_names):
    with tabs[i]:
        st.markdown(f"### {p}")
        render_portfolio_view(p)

# --- Admin controls (only if logged in) ---
if is_admin:
    st.markdown("---")
    st.subheader("Admin (edit enabled)")

    colA, colB = st.columns([1, 2])

    with colA:
        st.markdown("#### Create a new portfolio")
        new_name = st.text_input("Portfolio name", value="", placeholder="e.g., Long Only, Trading, IRA")
        if st.button("Add portfolio", type="primary"):
            name = (new_name or "").strip()
            if not name:
                st.error("Enter a portfolio name.")
            elif name in set(portfolio_names):
                st.warning("That portfolio already exists.")
            else:
                portfolios_df = pd.concat([portfolios_df, pd.DataFrame([{"portfolio": name}])], ignore_index=True)
                save_portfolios(portfolios_df)
                st.success("Portfolio added.")
                st.rerun()

    with colB:
        st.markdown("#### Add / Delete transactions")
        active_portfolio = st.selectbox("Active portfolio", portfolio_names, index=0)

        c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
        with c1:
            t_date = st.date_input("Transaction date", value=date.today(), key="admin_date")
        with c2:
            t_ticker = st.text_input("Ticker", value="AAPL", key="admin_ticker")
        with c3:
            t_side = st.selectbox("Buy/Sell", ["buy", "sell"], key="admin_side")
        with c4:
            t_shares = st.number_input(
                "Shares (3 decimals)",
                min_value=0.0,
                value=1.000,
                step=0.001,
                format="%.3f",
                key="admin_shares",
            )
        with c5:
            t_price = st.number_input(
                "Price",
                min_value=0.0,
                value=100.00,
                step=0.01,
                format="%.2f",
                key="admin_price",
            )

        if st.button("Save transaction", type="primary", key="save_txn"):
            row = {
                "txn_id": str(pd.Timestamp.utcnow().value),
                "portfolio": active_portfolio,
                "ticker": t_ticker.strip().upper(),
                "date": pd.to_datetime(t_date),
                "side": t_side,
                "shares": float(round(t_shares, 3)),
                "price": float(t_price),
            }
            txns_all = pd.concat([txns_all, pd.DataFrame([row])], ignore_index=True)
            save_txns(txns_all)
            st.success("Saved.")
            st.rerun()

        st.markdown("**Delete a transaction**")
        txns_active = txns_all[txns_all["portfolio"] == active_portfolio].copy()
        if txns_active.empty:
            st.info("No transactions in this portfolio.")
        else:
            disp = txns_active.copy()
            disp["display"] = (
                disp["date"].dt.date.astype(str)
                + " | "
                + disp["ticker"]
                + " | "
                + disp["side"]
                + " | "
                + disp["shares"].map(lambda x: f"{x:.3f}")
                + " @ "
                + disp["price"].map(lambda x: f"{x:.2f}")
                + " | id="
                + disp["txn_id"].astype(str)
            )
            choice = st.selectbox("Select transaction", disp["display"].tolist(), key="delete_choice")
            chosen_id = choice.split("id=")[-1].strip()

            if st.button("Delete selected", key="delete_btn"):
                txns_all = txns_all[txns_all["txn_id"].astype(str) != chosen_id].copy()
                save_txns(txns_all)
                st.warning("Deleted.")
                st.rerun()

st.caption("Note: This uses CSV files (transactions.csv + portfolios.csv). For production persistence, swap to SQLite/Postgres.")
