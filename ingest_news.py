"""Step 2: macro news via Tavily -> Chroma 'news' collection.

Design notes:
  * Two modes. Default = historical backfill in 14-day windows (gives the model
    a news history). --recent = last 7 days only (feeds the macro summary).
  * Short focused queries, one search per (query, window), as Tavily recommends.
  * Every raw response is cached, so re-runs cost no credits.
  * LOOK-AHEAD RULE: an article is only usable as a feature if we KNOW when it
    was published. Articles without a parseable published date are stored but
    flagged (has_date=0) and must be excluded from backtests.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
import config
import store
from ingest_apify import date_chunks

load_dotenv()
def parse_published(value):
    """Tavily gives RFC-2822 ('Mon, 05 Oct 2026 13:00:00 GMT') or ISO dates."""
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def cache_path(query: str, start: date, end: date, topic: str):
    h = hashlib.sha1(f"{query}|{start}|{end}|{topic}".encode()).hexdigest()[:10]
    return config.NEWS_DIR / f"{start}_{end}_{h}.json"
def search(client, query: str, start: date, end: date) -> list[dict]:
    """One cached Tavily call. Falls back to topic='general' if 'news' errors."""
    topic = config.NEWS_TOPIC
    cache = cache_path(query, start, end, topic)
    if cache.exists():
        return json.loads(cache.read_text())
    kwargs = dict(query=query, max_results=config.NEWS_MAX_RESULTS, search_depth="basic",
                  start_date=start.isoformat(), end_date=end.isoformat())
    try:
        resp = client.search(topic=topic, **kwargs)
    except Exception as err:
        print(f"  topic='{topic}' failed ({err}); retrying with 'general'")
        try:
            resp = client.search(topic="general", **kwargs)
        except Exception as err2:
            print(f"  skipped {query!r} {start}->{end}: {err2}")
            return []  # not cached, so a later run retries it

    results = resp.get("results", [])
    for r in results:
        r["_query"] = query
    config.NEWS_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(results, indent=2))
    time.sleep(0.3)  # be polite to the API
    return results
def normalise(r: dict):
    url, title = r.get("url"), (r.get("title") or "").strip()
    if not url or not title:
        return None
    pub = parse_published(r.get("published_date"))
    content = (r.get("content") or "").strip()
    return {
        "news_id": hashlib.sha1(url.encode()).hexdigest()[:16],
        "document": f"{title}. {content}",   # embedded text
        "meta": {
            "title": title,
            "url": url,
            "source": urlparse(url).netloc.removeprefix("www."),
            "query": r.get("_query"),
            "score": float(r["score"]) if r.get("score") is not None else None,
            "has_date": int(pub is not None),
            "published_utc": pub.isoformat() if pub else None,
            "pub_epoch": int(pub.timestamp()) if pub else None}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", action="store_true", help="last 7 days only")
    ap.add_argument("--yes", action="store_true", help="skip the credit confirmation")
    args = ap.parse_args()
    end = date.today() if args.recent else config.END_DATE
    start = end - timedelta(days=7 if args.recent else config.LOOKBACK_DAYS)
    windows = [(start, end)] if args.recent else list(date_chunks(start, end, config.NEWS_WINDOW_DAYS))
    calls = len(windows) * len(config.NEWS_QUERIES)
    todo = sum(not cache_path(q, a, b, config.NEWS_TOPIC).exists()
               for a, b in windows for q in config.NEWS_QUERIES)
    print(f"{calls} searches ({todo} not cached). Basic search = ~1 Tavily credit each.")
    if todo and not args.yes and input("Run? [y/N] ").strip().lower() != "y":
        raise SystemExit("Aborted, nothing spent.")
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    raw = [r for a, b in windows for q in config.NEWS_QUERIES for r in search(client, q, a, b)]
    articles = [n for n in map(normalise, raw) if n]
    unique = store.dedupe_by(articles, "news_id")
    undated = sum(1 for a in unique if not a["meta"]["has_date"])
    print(f"{len(raw)} results -> {len(unique)} unique articles ({undated} without a published date)")
    print(f"{store.upsert_news(unique)} articles in store.")
    df = store.news_df()
    if len(df) and df["has_date"].sum():
        d = df[df.has_date == 1]
        print("published range:", d.published_utc.min()[:10], "->", d.published_utc.max()[:10])
        print(d.source.value_counts().head(8).to_string())
if __name__ == "__main__":
    main()
