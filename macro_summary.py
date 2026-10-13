"""Step 7 (assignment item 2): one-page macro outlook.

Rule-based on purpose: every line traces to a scraped event or a cited headline,
nothing is invented, and it runs with no extra API key. Sections:
  sentiment read -> latest releases vs forecast -> upcoming releases -> news themes -> takeaway
"""
import argparse

import numpy as np
import pandas as pd

import config
import features as F

OUT = config.ROOT / "outputs"


def _direction(delta):
    if delta is None or np.isnan(delta):
        return "unclear"
    return "improving" if delta > 0.05 else "deteriorating" if delta < -0.05 else "broadly unchanged"


def _tone(x):
    return "unclear" if x is None or np.isnan(x) else "constructive" if x > 0.1 else "cautious" if x < -0.1 else "mixed"


def _verdict(actual, forecast):
    if actual is None or forecast is None or np.isnan(actual) or np.isnan(forecast):
        return "no consensus available"
    if abs(actual - forecast) < 1e-9:
        return "in line with forecast"
    return "above forecast" if actual > forecast else "below forecast"


def build(ev: pd.DataFrame, nw: pd.DataFrame, asof: pd.Timestamp) -> str:
    L = []
    d = nw[nw["has_date"] == 1].copy() if "has_date" in nw else nw.copy()
    d["pub"] = pd.to_datetime(d["published_utc"], utc=True)
    d["sent"] = d["text"].map(F.score_text)
    anchor = min(asof, d.pub.max())                       # don't pretend news is newer than it is

    e = ev.copy()
    e["ts"] = pd.to_datetime(e["ts_utc"], utc=True)
    last_ev = e.ts.max()
    L.append(f"# US Macro Outlook (as of {asof:%d %b %Y})")
    L.append(f"*Calendar data through {last_ev:%d %b %Y}; news through {d.pub.max():%d %b %Y}. "
             f"Generated from scraped data by fixed rules.*")
    if last_ev < asof - pd.Timedelta(days=14):
        L.append(f"> **Note:** the event calendar ends {(asof - last_ev).days} days before this date, "
                 f"so releases since {last_ev:%d %b} are not reflected in the data section.")

    method = "FinBERT" if F._backend()[0] == "finbert" else "a simple word list"
    # --- 1. sentiment read
    w1 = d[(d.pub >= anchor - pd.Timedelta(days=7)) & (d.pub <= anchor)].sent.mean()
    w0 = d[(d.pub >= anchor - pd.Timedelta(days=14)) & (d.pub < anchor - pd.Timedelta(days=7))].sent.mean()
    w4 = d[(d.pub >= anchor - pd.Timedelta(days=28)) & (d.pub <= anchor)].sent.mean()
    L.append("\n## Sentiment")
    L.append(f"News tone over the last week is **{_tone(w1)}** (score {w1:+.2f}), {_direction(w1 - w0)} "
             f"versus the prior week ({w0:+.2f}); the 4-week average is {w4:+.2f}. "
             f"Scores come from {method} applied to headlines and snippets; treat them as a rough gauge.")

    # --- 2. latest releases: the three the assignment names first (CPI, FOMC, employment),
    #        then other high-impact releases to fill the list.
    done = e[e.actual.notna()].sort_values("ts", ascending=False)
    nm = done.name.str.lower()
    picks = []
    for mask in (done.category == "CPI", done.category == "FOMC_DECISION",
                 nm.str.contains("nonfarm"), nm.str.contains("unemployment rate")):
        if mask.any():
            picks.append(done[mask].iloc[0])
    seen = {r["name"] for r in picks}
    for _, r in done[done.importance == "high"].iterrows():
        if len(picks) >= 6:
            break
        if r["name"] not in seen:
            picks.append(r)
            seen.add(r["name"])
    L.append("\n## Latest releases (CPI, Fed decision and jobs included)")
    if picks:
        for r in sorted(picks, key=lambda r: r["ts"], reverse=True):
            fc = r["forecast_raw"] if isinstance(r["forecast_raw"], str) else None
            tail = f" vs forecast {fc}, {_verdict(r['actual'], r['forecast'])}" if fc else " (no consensus listed)"
            L.append(f"- {r['ts']:%d %b}: **{r['name']}**: actual {r['actual_raw']}{tail}")
        sur = e[(e.importance == "high") & e.surprise.notna() & (e.ts >= last_ev - pd.Timedelta(days=30))]
        if len(sur):
            up_n, dn_n = int((sur.surprise > 0).sum()), int((sur.surprise < 0).sum())
            L.append(f"\nOf {len(sur)} high-impact releases with a consensus in the final 30 days of data, "
                     f"{up_n} came in above forecast, {dn_n} below and {len(sur) - up_n - dn_n} in line.")
    else:
        L.append("- No releases with actual values in the data.")

    # --- 3. upcoming
    up = e[(e.ts > asof) & (e.ts <= asof + pd.Timedelta(days=14)) & (e.importance == "high")].sort_values("ts")
    if len(up):
        L.append("\n## Scheduled in the next two weeks")
        for r in up.head(6).itertuples():
            L.append(f"- {r.ts:%d %b}: {r.name}")

    # --- 4. news themes
    recent = d[d.pub >= anchor - pd.Timedelta(days=14)]
    L.append("\n## What the news is saying (last two weeks)")
    for topic, g in recent.groupby("query"):
        top = g.sort_values(["score", "pub"], ascending=False).head(2)
        L.append(f"**{topic}**")
        for r in top.itertuples():
            tag = "+" if r.sent > 0.15 else "-" if r.sent < -0.15 else "~"
            L.append(f"- [{tag}] {r.title[:110]}: {r.source}, {r.pub:%d %b} ({r.url})")

    # --- 5. takeaway
    L.append("\n## Takeaway")
    L.append(f"Headline tone is {_tone(w1)} and {_direction(w1 - w0)}. "
             f"This is an automated read of scraped data, not a forecast or investment advice; "
             f"check the cited articles before relying on any line above.")
    return "\n".join(L)


def main():
    import store
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", help="YYYY-MM-DD (default: today, UTC)")
    args = ap.parse_args()
    asof = pd.Timestamp(args.asof, tz="UTC") if args.asof else pd.Timestamp.now(tz="UTC")
    md = build(store.events_df(), store.news_df(), asof)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "macro_outlook.md").write_text(md, encoding="utf-8")
    print(md)
    print("\n->", OUT / "macro_outlook.md")


if __name__ == "__main__":
    main()