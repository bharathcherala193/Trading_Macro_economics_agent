# Macro-Event Strategy Selection Pipeline

Predicts which trading-strategy parameter set (permutation) will perform best in the coming week,
using macro-calendar events, macro news sentiment and market state, validated walk-forward.

## Setup
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    copy .env.example .env      # then add your own keys
Optional, for better sentiment: `pip install transformers torch` (FinBERT). Otherwise a word list is used.

## Run
    python run_all.py                 # scrapes (asks before spending credits), then builds everything
    python run_all.py --skip-ingest   # reuse cached data in data/raw and data/raw_news

## Steps
| Script | Purpose | Output |
|---|---|---|
| ingest_apify.py | Macro events via Apify (UTC + EST times, actual, forecast) | Chroma collection |
| ingest_news.py | Recent macro news via Tavily | Chroma collection |
| trades.py | Trade logs (none provided, so generated: MA crossover on SPY) | data/trades.csv |
| features.py, walkforward.py | Features, model, walk-forward validation | data/walkforward_*.csv |
| macro_summary.py | 1-page macro outlook | outputs/macro_outlook.md |
| charts.py | Sentiment over time charts | outputs/*.png |
| report.py | Sharpe, Sortino, max drawdown report | EVALUATION_REPORT.md |

## Reproducibility
- Apify actor: `solidcode/economic-calendar-data-scraper` (filters in config.py). Tavily topic: `news`.
- Raw responses are cached in data/raw and data/raw_news, so re-runs cost nothing.
- Never commit `.env`. Keys are read from APIFY_TOKEN and TAVILY_API_KEY.

## Scope
Trade logs are self-generated and the sample is short; read results with the uncertainty stated in EVALUATION_REPORT.md.