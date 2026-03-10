# portfolio-management-tool

## Company bucket classification script

Use `scripts/classify_companies.py` to automatically map tickers into your requested buckets while keeping Yahoo industry detail.

Buckets:
- Infrastructure
- Real Estate
- Technology
- Media & Telecommunications
- Consumer & Retail
- Healthcare
- Natural Resources & Energy
- Industrials
- Financial Institutions

Examples:

```bash
python scripts/classify_companies.py --tickers AAPL AMZN VZ PFE XOM
python scripts/classify_companies.py --input holdings.csv --output classified_companies.csv
```

Input CSV should include a `ticker` column.
