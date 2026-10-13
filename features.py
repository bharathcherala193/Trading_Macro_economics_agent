"""Step 4: build the (permutation x week) feature table.

One row = one strategy permutation in one week.
  target y      = total net return of the trades ENTERED that week (0 if flat)
  label_known_ts = when the last of those trades has exited, i.e. when y is
                  fully realised. Walk-forward training must respect this.

LOOK-AHEAD RULES (every feature for week w may only use information that exists
at the START of w):
  * market / news / past-event surprises : strictly before w
  * the event SCHEDULE (what is due this week, days to next CPI) : allowed,
    because calendars are published in advance. Their ACTUAL values are not.
  * permutation history : only trades that already EXITED before w
"""
import atexit
import hashlib
import json
import re

import numpy as np
import pandas as pd

import config

WEEK = pd.Timedelta(days=7)
CATS = ["CPI", "FOMC_DECISION", "FED_COMMS", "EMPLOYMENT", "INFLATION_OTHER", "GROWTH"]

# ---- crude, transparent sentiment (v1). Swap for FinBERT / an LLM later. ----
POS = {"rally", "rallies", "gain", "gains", "rise", "rises", "rose", "beat", "beats", "strong", "growth",
       "surge", "surges", "optimism", "upbeat", "cools", "cooling", "eases", "easing", "cut", "cuts",
       "rebound", "improves", "boost", "record", "resilient", "soft landing"}
NEG = {"fall", "falls", "fell", "drop", "drops", "slump", "plunge", "selloff", "sell-off", "recession",
       "fears", "weak", "weaker", "miss", "misses", "hot", "hotter", "sticky", "hike", "hikes", "tightening",
       "layoffs", "concern", "concerns", "worries", "slowdown", "downturn", "tumble", "crash", "risk-off",
       "uncertainty", "inflation worries", "stagflation"}


def _lexicon_score(text: str) -> float:
    t = str(text).lower()
    toks = re.findall(r"[a-z\-']+", t)
    pos = sum(w in POS for w in toks) + sum(p in t for p in ("soft landing", "record high"))
    neg = sum(w in NEG for w in toks) + sum(p in t for p in ("inflation worries", "sell-off"))
    return (pos - neg) / (pos + neg) if pos + neg else 0.0


# ---- sentiment backend -------------------------------------------------------
# FinBERT (ProsusAI/finbert) understands phrases like "worse than expected" and
# negation, which a word list cannot. It needs `pip install transformers torch`
# (large download, first run only). If it is missing we fall back to the word list
# and say so. Scores are cached on disk, keyed by backend + text, so FinBERT runs
# once per article, not once per pipeline run.
_BACKEND = None
_CACHE = None
CACHE_PATH = config.ROOT / "data" / "sentiment_cache.json"


def _backend():
    global _BACKEND
    if _BACKEND is None:
        try:
            from transformers import pipeline
            _BACKEND = ("finbert", pipeline("text-classification", model="ProsusAI/finbert", top_k=None))
            print("sentiment: using FinBERT")
        except Exception as err:
            _BACKEND = ("lexicon", None)
            print(f"sentiment: FinBERT unavailable ({type(err).__name__}); using the crude word list. "
                  f"For better scores: pip install transformers torch")
    return _BACKEND


def _save_cache():
    if _CACHE is not None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(_CACHE))


def score_text(text: str) -> float:
    """Sentiment in [-1, 1]: P(positive) - P(negative) for FinBERT."""
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
        atexit.register(_save_cache)
    name, clf = _backend()
    key = f"{name}:{hashlib.sha1(str(text).encode()).hexdigest()}"
    if key in _CACHE:
        return _CACHE[key]
    if name == "finbert":
        res = clf(str(text)[:2000], truncation=True, max_length=512)
        res = res[0] if isinstance(res[0], list) else res      # output shape differs by version
        p = {d["label"].lower(): d["score"] for d in res}
        val = float(p.get("positive", 0.0) - p.get("negative", 0.0))
    else:
        val = _lexicon_score(text)
    _CACHE[key] = val
    return val


# ------------------------------------------------------------ feature blocks
def market_feats(prices, w):
    p = prices[(prices.index >= w - WEEK) & (prices.index < w)]
    if len(p) < 5:
        return {"prev_week_ret": np.nan, "prev_week_vol": np.nan, "prev_week_range": np.nan}
    r = p.close.pct_change().dropna()
    return {"prev_week_ret": p.close.iloc[-1] / p.open.iloc[0] - 1,
            "prev_week_vol": r.std(),
            "prev_week_range": (p.high.max() - p.low.min()) / p.close.iloc[0]}


def event_feats(ev, w):
    out = {}
    due = ev[(ev.ts >= w) & (ev.ts < w + WEEK)]            # schedule: known in advance
    for cat in CATS:
        out[f"n_{cat.lower()}"] = int((due.category == cat).sum())
    out["n_high"] = int((due.importance == "high").sum())
    for cat, key in (("CPI", "cpi"), ("FOMC_DECISION", "fomc"), ("EMPLOYMENT", "emp")):
        c = ev[ev.category == cat]
        before, after = c[c.ts < w], c[c.ts >= w]
        out[f"days_since_{key}"] = (w - before.ts.max()).days if len(before) else np.nan
        out[f"days_to_{key}"] = (after.ts.min() - w).days if len(after) else np.nan
        s = before.surprise.dropna()                        # only PAST actuals
        out[f"last_{key}_surprise_sign"] = float(np.sign(s.iloc[-1])) if len(s) else np.nan
    s28 = ev[(ev.ts < w) & (ev.ts >= w - 4 * WEEK)].surprise.dropna()
    out["surprise_idx_28d"] = float(np.sign(s28).mean()) if len(s28) else np.nan
    return out


def news_feats(nw, w):
    a = nw[(nw.pub >= w - WEEK) & (nw.pub < w)]
    b = nw[(nw.pub >= w - 2 * WEEK) & (nw.pub < w - WEEK)]
    c = nw[(nw.pub >= w - 4 * WEEK) & (nw.pub < w)]
    s7 = a.sent.mean() if len(a) else np.nan
    return {"sent_7d": s7, "sent_28d": c.sent.mean() if len(c) else np.nan,
            "sent_delta": s7 - b.sent.mean() if len(a) and len(b) else np.nan, "news_n_7d": len(a)}


def perm_feats(tp, w):
    done = tp[tp.exit_ts < w]                               # only trades that already exited
    return {"perm_pnl_4w": done[done.exit_ts >= w - 4 * WEEK].ret.sum(),
            "perm_mean_ret": done.ret.mean() if len(done) else np.nan,
            "perm_win_rate": (done.ret > 0).mean() if len(done) else np.nan,
            "perm_n_done": len(done)}


# ------------------------------------------------------------------ table ---
def build_table(events, news, trades, prices, through=None) -> pd.DataFrame:
    ev = events.copy()
    ev["ts"] = pd.to_datetime(ev["ts_utc"], utc=True)
    for c in ("surprise", "importance", "category"):
        if c not in ev:
            ev[c] = np.nan
    ev = ev.sort_values("ts")

    nw = news[news.get("has_date", 1) == 1].copy() if len(news) else news.copy()
    if len(nw):
        nw["pub"] = pd.to_datetime(nw["published_utc"], utc=True)
        nw["sent"] = nw["text"].map(score_text)
    else:
        nw = pd.DataFrame({"pub": pd.to_datetime([], utc=True), "sent": []})

    tr = trades.copy()
    tr["entry_ts"] = pd.to_datetime(tr.entry_ts, utc=True)
    tr["exit_ts"] = pd.to_datetime(tr.exit_ts, utc=True)
    t0 = tr.entry_ts.min()
    start = t0.normalize() - pd.Timedelta(days=t0.weekday())
    end = tr.entry_ts.max()
    if through is not None:
        end = min(end, pd.Timestamp(through, tz="UTC"))
    weeks = [w for w in pd.date_range(start, end, freq="7D") if w + WEEK <= end + pd.Timedelta(days=1)]

    tr["week"] = tr.entry_ts.dt.normalize() - pd.to_timedelta(tr.entry_ts.dt.weekday, unit="D")
    agg = tr.groupby(["perm_id", "week"]).agg(y=("ret", "sum"), n_trades=("ret", "size"),
                                             label_known_ts=("exit_ts", "max"))
    params = tr.drop_duplicates("perm_id").set_index("perm_id")[["fast", "slow", "stop_loss", "take_profit"]]
    by_perm = {p: g for p, g in tr.groupby("perm_id")}

    rows = []
    for w in weeks:
        wk = {**market_feats(prices, w), **event_feats(ev, w), **news_feats(nw, w),
              "month": w.month, "week_of_month": (w.day - 1) // 7 + 1}
        for pid in params.index:
            a = agg.loc[(pid, w)] if (pid, w) in agg.index else None
            rows.append({"perm_id": pid, "week": w,
                         "y": a.y if a is not None else 0.0,
                         "n_trades": int(a.n_trades) if a is not None else 0,
                         "label_known_ts": a.label_known_ts if a is not None else w + WEEK,
                         **params.loc[pid].to_dict(),
                         "fast_over_slow": params.loc[pid, "fast"] / params.loc[pid, "slow"],
                         **perm_feats(by_perm[pid], w), **wk})
    return pd.DataFrame(rows)


NON_FEATURES = {"perm_id", "week", "y", "n_trades", "label_known_ts"}


def feature_cols(tbl):
    return [c for c in tbl.columns if c not in NON_FEATURES]