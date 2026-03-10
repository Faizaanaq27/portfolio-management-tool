#!/usr/bin/env python3
"""Classify companies into portfolio sector buckets, preserving industry detail.

Usage examples:
  python scripts/classify_companies.py --tickers AAPL MSFT VZ XOM
  python scripts/classify_companies.py --input holdings.csv --output classified.csv

Input CSV must include a `ticker` column.
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

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


def fetch_sector_industry(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in sorted(set(tickers)):
        try:
            info = yf.Ticker(ticker).info or {}
            sector = info.get("sector") or "Unknown"
            industry = info.get("industry") or "Unknown"
        except Exception:
            sector, industry = "Unknown", "Unknown"
        rows.append({"ticker": ticker, "sector": sector, "industry": industry})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify companies into sector buckets + industry")
    parser.add_argument("--input", type=Path, help="CSV with a ticker column")
    parser.add_argument("--output", type=Path, help="Optional output CSV path")
    parser.add_argument("--tickers", nargs="*", default=[], help="Ticker list, e.g. AAPL MSFT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tickers = [str(t).upper().strip() for t in args.tickers if str(t).strip()]
    if args.input:
        df = pd.read_csv(args.input)
        if "ticker" not in df.columns:
            raise ValueError("Input CSV must include a `ticker` column")
        csv_tickers = df["ticker"].astype(str).str.upper().str.strip().tolist()
        tickers.extend([t for t in csv_tickers if t])

    tickers = sorted(set(tickers))
    if not tickers:
        raise ValueError("Provide at least one ticker via --tickers or --input")

    out = fetch_sector_industry(tickers)
    out["bucket"] = [classify_sector_bucket(s, i) for s, i in zip(out["sector"], out["industry"])]
    out = out[["ticker", "bucket", "sector", "industry"]].sort_values(["bucket", "industry", "ticker"])

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Saved {len(out)} rows to {args.output}")
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
