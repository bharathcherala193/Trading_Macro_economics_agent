"""Storage layer on ChromaDB (local, persistent, no server).

Why this shape:
  * Every record gets a text `document` + numeric/string `metadata`.
    - the document is what gets embedded (semantic search / RAG later),
    - the metadata is what we filter on and turn into model features.
  * Chroma can only range-filter NUMBERS, so timestamps are stored as epoch
    seconds (`ts_epoch`) next to the readable ISO string.
  * Chroma metadata can't hold None, so missing values are simply left out.
  * upsert() keyed by a stable id => re-running ingestion never duplicates.

Honest limit: Chroma is great for events + news text, but it has no joins or
GROUP BY. Heavy trade-log work should happen in pandas after export (see
events_df). That's why trades are NOT stored here.
"""
import chromadb
import pandas as pd
import config

EMBED_FN = None
_client = None

def _collection(name: str):
    global _client
    if _client is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    kwargs = {"embedding_function": EMBED_FN} if EMBED_FN else {}
    return _client.get_or_create_collection(name, **kwargs)

def _clean(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if v is not None}

def dedupe(events: list[dict]) -> list[dict]:
    """Chroma rejects a batch containing the same id twice. Overlapping scrape
    windows produce exactly that, so keep one record per id (the last seen)."""
    return list({e["event_id"]: e for e in events}.values())

def upsert_events(events: list[dict]) -> int:
    col = _collection("events")
    events = dedupe(events)
    for i in range(0, len(events), 500):  # batch to stay under Chroma's max batch size
        chunk = events[i:i + 500]
        col.upsert(
            ids=[e["event_id"] for e in chunk],
            documents=[e["document"] for e in chunk],
            metadatas=[_clean(e["meta"]) for e in chunk],
        )
    return col.count()

def events_df() -> pd.DataFrame:
    """All events as a DataFrame, sorted by time. This is the bridge to the
    feature-engineering / modelling code."""
    res = _collection("events").get(include=["metadatas"])
    df = pd.DataFrame(res["metadatas"])
    df.insert(0, "event_id", res["ids"])
    return df.sort_values("ts_epoch").reset_index(drop=True) if len(df) else df

def search_events(query: str, n: int = 5, category: str | None = None):
    """Semantic search, optionally restricted by category."""
    where = {"category": category} if category else None
    return _collection("events").query(query_texts=[query], n_results=n, where=where)

def upsert_news(articles: list[dict]) -> int:
    col = _collection("news")
    articles = dedupe_by(articles, "news_id")
    for i in range(0, len(articles), 500):
        chunk = articles[i:i + 500]
        col.upsert(
            ids=[a["news_id"] for a in chunk],
            documents=[a["document"] for a in chunk],
            metadatas=[_clean(a["meta"]) for a in chunk],
        )
    return col.count()

def dedupe_by(items: list[dict], key: str) -> list[dict]:
    return list({i[key]: i for i in items}.values())

def news_df() -> pd.DataFrame:
    res = _collection("news").get(include=["metadatas", "documents"])
    df = pd.DataFrame(res["metadatas"])
    if not len(df):
        return df
    df.insert(0, "news_id", res["ids"])
    df["text"] = res["documents"]
    return df

def search_news(query: str, n: int = 8, since_epoch: int | None = None):
    """Semantic search over news; since_epoch restricts to recent articles."""
    where = {"pub_epoch": {"$gte": since_epoch}} if since_epoch else None
    return _collection("news").query(query_texts=[query], n_results=n, where=where)