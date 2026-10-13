"""Central config. Everything you might want to change lives here."""
from datetime import date, timedelta
from pathlib import Path

_here = Path(__file__).resolve().parent
ROOT = _here.parent if _here.name == "src" else _here
RAW_DIR = ROOT / "data" / "raw"
CHROMA_DIR = ROOT / "data" / "chroma"
APIFY_ACTOR_ID = "solidcode/economic-calendar-data-scraper"  # Investing.com calendar, $8 / 1,000 rows
LOOKBACK_DAYS = 180
CHUNK_DAYS = 30
END_DATE = date(2026, 10, 9)
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAYS)
ACTOR_FILTERS = {
       "countries": ["united states"],
       "importances": ["high"],
       "categories": ["inflation", "employment", "central_banks"],   # CPI, jobs, Fed
       "timeZone": "GMT",
   }
MAX_RESULTS_PER_CHUNK = 120   # hard cost cap per run; a warning prints if it's hit
PRICE_PER_1K_ROWS = 8.0         # USD, from the actor's pricing page

SOURCE_TZ = "UTC"

CATEGORY_KEYWORDS = [
    ("FED_COMMS", ["speaks", "speech", "testifies", "minutes", "beige book", "fed chair", "fomc member"]),
    ("FOMC_DECISION", ["fed interest rate", "federal funds rate", "interest rate decision", "fomc statement", "fomc press"]),
    ("CPI", ["cpi", "consumer price"]),
    ("INFLATION_OTHER", ["pce", "ppi", "producer price", "import price", "inflation"]),
    ("EMPLOYMENT", ["nonfarm", "non-farm", "unemployment", "jobless", "adp", "jolts", "hourly earnings",
                    "employment change", "participation rate", "payroll"]),
    ("GROWTH", ["gdp", "retail sales", "ism", "pmi", "industrial production", "durable goods",
                "consumer confidence", "consumer sentiment", "michigan", "housing", "building permits",
                "home sales", "trade balance", "factory orders"]),]


NEWS_DIR = ROOT / "data" / "raw_news"
NEWS_WINDOW_DAYS = 14          # one search per (query, 14-day window)
NEWS_MAX_RESULTS = 10
NEWS_TOPIC = "news"            # the 'news' topic returns published_date; falls back to 'general'
NEWS_QUERIES = [               # short, focused queries work best with Tavily
    "Federal Reserve interest rate decision",
    "US CPI inflation report",
    "US jobs report nonfarm payrolls",
    "US economy outlook recession",
    "US Treasury yields dollar"]

TRADE_SYMBOL = "SPY"            # US macro events move US equities; liquid; free data
TRADE_INTERVAL = "1h"           # yfinance serves 1h bars for ~730 days, plenty for 180
COST_BPS = 1.0                  # per side, in basis points (spread + slippage assumption)
TRADES_CSV = ROOT / "data" / "trades.csv"
PRICES_CSV = ROOT / "data" / "prices.csv"

PARAM_GRID = {
    "fast": [5, 10, 20],
    "slow": [30, 50, 100],
    "stop_loss": [0.005, 0.01],     # 0.5%, 1%
    "take_profit": [0.01, 0.02],    # 1%, 2%
}                                    # 3*3*2*2 = 36 permutations
