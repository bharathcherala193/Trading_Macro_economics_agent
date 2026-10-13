"""Step 1: pull ~180 days of macro events via Apify and load them into SQLite.

Design, in the order I'd think about it:
  1. Scrape in 30-day chunks. One giant run is slow, and if it fails you pay again.
  2. Cache every raw response to disk. Re-running costs $0 and reviewers can
     reproduce without burning credits.
  3. Normalise defensively. Community actors name fields differently and can
     change them, so we try several key names instead of hard-coding one.
  4. Convert to UTC immediately. Everything downstream assumes UTC.
"""
import argparse
import hashlib
import json
import os
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from apify_client import ApifyClient
from apify_client.errors import ForbiddenError
from dotenv import load_dotenv
import config
import store

load_dotenv()

def date_chunks(start: date, end: date, size: int):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=size - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)

def build_input(start: date, end: date) -> dict:
    return {
        **config.ACTOR_FILTERS,
        "fromDate": start.isoformat(),
        "toDate": end.isoformat(),
        "maxResults": config.MAX_RESULTS_PER_CHUNK}


def cache_path(run_input: dict):
    
    h = hashlib.sha1(json.dumps(run_input, sort_keys=True).encode()).hexdigest()[:8]
    return config.RAW_DIR / f"{run_input['fromDate']}_{run_input['toDate']}_{h}.json"

def fetch_chunk(client: ApifyClient, start: date, end: date) -> list[dict]:
    run_input = build_input(start, end)
    cache = cache_path(run_input)
    if cache.exists():  # cache hit = no credits spent
        return json.loads(cache.read_text())
    run = client.actor(config.APIFY_ACTOR_ID).call(run_input=run_input)
    dataset_id = field(run, "default_dataset_id", "defaultDatasetId")
    run_id = field(run, "id")
    items = list(client.dataset(dataset_id).iterate_items())

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items, indent=2, default=str))
    print(f"  fetched {len(items):>4} rows for {start} -> {end} (run {run_id})")
    if len(items) >= config.MAX_RESULTS_PER_CHUNK:
        print(f"  WARNING: hit the {config.MAX_RESULTS_PER_CHUNK}-row cap; this window is "
              f"probably truncated. Raise MAX_RESULTS_PER_CHUNK or shrink CHUNK_DAYS.")
    return items

def load_all_cached() -> list[dict]:
    """Offline mode: read every cached chunk, spend nothing."""
    items = []
    for f in sorted(config.RAW_DIR.glob("*.json")):
        items.extend(json.loads(f.read_text()))
    return items

def keep(raw: dict) -> bool:
    """Apply the same filters locally, so cached rows scraped WITHOUT filters
    (e.g. your first 3 runs) are narrowed to what we actually want."""
    countries = {c.lower() for c in config.ACTOR_FILTERS.get("countries", [])}
    imps = {i.lower() for i in config.ACTOR_FILTERS.get("importances", [])}
    if countries and str(raw.get("country", "")).lower() not in countries:
        return False
    if imps and str(raw.get("importance", "")).lower() not in imps:
        return False
    return True

def field(obj, *names):
    """Read a field from a dict or an attribute-style object."""
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v is not None:
            return v
    raise KeyError(names)

def pick(d: dict, *keys):
    """Return the first non-empty value among several possible field names."""
    for k in keys:
        if d.get(k) not in (None, "", "-"):
            return d[k]
    return None

_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

def to_number(s):
    """'3.2%' -> 3.2 ; '225K' -> 225000 ; '-0.1' -> -0.1 ; junk -> None.
    The unit multiplier matters: actual and forecast must be on the same
    scale, otherwise 'surprise' is garbage."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    text = str(s).strip()
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not m:
        return None
    val = float(m.group())
    suffix = re.search(r"([KMBT])\s*$", text, re.I)
    return val * _MULT[suffix.group(1).upper()] if suffix else val

def categorise(name: str) -> str:
    low = name.lower()
    for cat, kws in config.CATEGORY_KEYWORDS:
        if any(k in low for k in kws):
            return cat
    return "OTHER"

def to_utc(raw: dict):
    """Build a timezone-aware UTC datetime. Prefer a full timestamp if the
    actor gives one; otherwise combine date + time using SOURCE_TZ."""
    full = pick(raw, "timestamp", "datetime", "dateTime", "date_time")
    if full:
        dt = datetime.fromisoformat(str(full).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(config.SOURCE_TZ))
        return dt.astimezone(ZoneInfo("UTC"))

    d, t = pick(raw, "date", "event_date"), pick(raw, "time", "event_time") or "00:00"
    if not d:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M"):
        try:
            dt = datetime.strptime(f"{str(d)[:10]} {t}", fmt)
            return dt.replace(tzinfo=ZoneInfo(config.SOURCE_TZ)).astimezone(ZoneInfo("UTC"))
        except ValueError:
            continue
    return None  

def normalise(raw: dict):
    name = pick(raw, "event", "eventName", "event_name", "title", "eventTitle")
    ts = to_utc(raw)
    if not name or ts is None:
        return None
    country = pick(raw, "country", "currency") or ""
    actual_raw = pick(raw, "actual")
    forecast_raw = pick(raw, "forecast")
    actual, forecast = to_number(actual_raw), to_number(forecast_raw)
    eid = hashlib.sha1(f"{ts.isoformat()}|{country}|{name}".encode()).hexdigest()[:16]
    category = categorise(name)
    surprise = None if actual is None or forecast is None else actual - forecast
    doc = (f"{name} ({country}) on {ts.date()}: actual {actual_raw or 'n/a'} "
           f"vs forecast {forecast_raw or 'n/a'}")
    return {
        "event_id": eid,
        "document": doc,  # this text is what gets embedded
        "meta": {
            "ts_utc": ts.isoformat(),
            "ts_epoch": int(ts.timestamp()),  # numeric => range-filterable
            "ts_est": ts.astimezone(ZoneInfo("America/New_York")).isoformat(),
            "country": country,
            "name": name,
            "category": category,
            "importance": str(pick(raw, "importance", "impact") or ""),
            "actual": actual,
            "forecast": forecast,
            "previous": to_number(pick(raw, "previous")),
            "actual_raw": None if actual_raw is None else str(actual_raw),
            "forecast_raw": None if forecast_raw is None else str(forecast_raw),
            "surprise": surprise,},}

def collect_raw(cached_only: bool, assume_yes: bool) -> list[dict]:
    if cached_only:
        return load_all_cached()
    chunks = list(date_chunks(config.START_DATE, config.END_DATE, config.CHUNK_DAYS))
    todo = [c for c in chunks if not cache_path(build_input(*c)).exists()]
    worst = len(todo) * config.MAX_RESULTS_PER_CHUNK * config.PRICE_PER_1K_ROWS / 1000
    print(f"{len(chunks)} chunks, {len(todo)} not cached. Worst-case cost: ${worst:.2f} "
          f"(you pay per row returned; US high/medium is usually far below the cap).")
    if todo and not assume_yes and input("Run the scrape? [y/N] ").strip().lower() != "y":
        raise SystemExit("Aborted, nothing spent.")
    client = ApifyClient(os.environ["APIFY_TOKEN"])
    raws = []
    for s, e in chunks:
        try:
            raws.extend(fetch_chunk(client, s, e))
        except ForbiddenError as err:
            print(f"Apify refused the run ({err}). Falling back to cached data only.")
            return load_all_cached()
    return raws

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached-only", action="store_true", help="use data/raw only, no Apify calls")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()
    raws = collect_raw(args.cached_only, args.yes)
    kept = [r for r in raws if keep(r)]
    rows = [n for n in map(normalise, kept) if n]
    print(f"{len(raws)} raw rows -> {len(kept)} after filters -> {len(rows)} parsed "
          f"({len(kept) - len(rows)} skipped: unparseable or 'All Day' time)")
    unique = store.dedupe(rows)
    print(f"{len(rows) - len(unique)} duplicate rows dropped (overlapping windows) -> {len(unique)} unique events")
    total = store.upsert_events(unique)  # upsert on id => safe to re-run
    print(f"{total} events in store.")
    df = store.events_df()
    if len(df):
        print(df["category"].value_counts().to_string())
        print("date range:", df["ts_utc"].min()[:10], "->", df["ts_utc"].max()[:10])
if __name__ == "__main__":
    main()