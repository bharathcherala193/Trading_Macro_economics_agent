"""Step 6 (assignment item 4): charts showing how macro sentiment changes over time.

Two outputs:
  1. sentiment_timeline.png : daily + 7-day rolling news sentiment, with the CPI /
     FOMC / payrolls release dates marked, and article volume underneath.
  2. sentiment_matrix.png   : the "matrix" view. Rows = news topic, columns = week,
     colour = mean sentiment. Shows WHICH topic drove a shift, not just that one happened.
Sentiment is the transparent word-list score in features.score_text (see report).
"""
import matplotlib
matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config
import features as F
OUT = config.ROOT / "outputs"
COLORS = {"CPI": "#c0392b", "FOMC decision": "#8e44ad", "Nonfarm payrolls": "#16a085"}

def method_label() -> str:
    return "FinBERT" if F._backend()[0] == "finbert" else "word-list"

def prep_news(nw: pd.DataFrame) -> pd.DataFrame:
    d = nw[nw["has_date"] == 1].copy() if "has_date" in nw else nw.copy()
    d["pub"] = pd.to_datetime(d["published_utc"], utc=True)
    d["sent"] = d["text"].map(F.score_text)
    d["day"] = d["pub"].dt.tz_convert(None).dt.normalize()  # naive UTC dates plot cleanly
    return d

def marker_dates(ev: pd.DataFrame):
    """Release dates worth marking. We mark nonfarm payrolls by NAME, not the whole
    EMPLOYMENT category, because that category also holds weekly jobless claims."""
    if ev is None or not len(ev):
        return {}, None
    e = ev.copy()
    e["day"] = pd.to_datetime(e["ts_utc"], utc=True).dt.tz_convert(None).dt.normalize()
    names = e["name"].str.lower()
    marks = {"CPI": sorted(e[e.category == "CPI"].day.unique()),
             "FOMC decision": sorted(e[e.category == "FOMC_DECISION"].day.unique()),
             "Nonfarm payrolls": sorted(e[names.str.contains("nonfarm")].day.unique())}
    return marks, e.day.max()

def sentiment_timeline(ev, nw, path):
    d = prep_news(nw)
    days = pd.date_range(d.day.min(), d.day.max(), freq="D")
    daily = d.groupby("day").sent.mean().reindex(days)
    counts = d.groupby("day").size().reindex(days, fill_value=0)
    sums = d.groupby("day").sent.sum().reindex(days, fill_value=0.0)
    # Article-weighted 7-day mean: a day with 8 articles counts for more than a day with 1.
    n7 = counts.rolling(7, min_periods=1).sum()
    roll = (sums.rolling(7, min_periods=1).sum() / n7).where(n7 >= 3)
    marks, last_ev = marker_dates(ev)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.scatter(days, daily, s=10, alpha=0.35, color="#7f8c8d", label="daily mean")
    a1.plot(days, roll, lw=2.2, color="#1f4e79", label="7-day rolling mean")
    a1.axhline(0, color="black", lw=0.8)
    for label, dates in marks.items():
        for i, dt in enumerate(dates):
            a1.axvline(dt, color=COLORS[label], alpha=0.35, lw=1, label=label if i == 0 else None)
    if last_ev is not None and last_ev < days.max():
        a1.axvspan(last_ev, days.max(), color="#bbbbbb", alpha=0.2, label="no calendar-event data")
    a1.set_ylabel("sentiment (-1 bearish ... +1 bullish)")
    a1.set_title(f"US macro news sentiment over time ({method_label()} score)")
    lo, hi = daily.min(), daily.max()
    a1.set_ylim(lo - 0.05 * (hi - lo), hi + 0.30 * (hi - lo))   # headroom so the legend covers no data
    a1.legend(loc="upper left", ncol=3, fontsize=8)
    a2.bar(days, counts, color="#95a5a6", width=0.9)
    a2.set_ylabel("articles / day")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def sentiment_matrix(nw, path):
    d = prep_news(nw)
    if "query" not in d:
        d["query"] = "all news"
    d["week"] = d["day"] - pd.to_timedelta(d["day"].dt.weekday, unit="D")
    pv = d.pivot_table(index="query", columns="week", values="sent", aggfunc="mean")
    pv = pv.reindex(columns=pd.date_range(pv.columns.min(), pv.columns.max(), freq="7D"))
    vlim = max(0.3, float(np.nanmax(np.abs(pv.values))))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eeeeee")  # grey = no articles that week for that topic

    fig, ax = plt.subplots(figsize=(13, 0.9 * len(pv) + 2.4))
    im = ax.imshow(np.ma.masked_invalid(pv.values), aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_yticks(range(len(pv)))
    ax.set_yticklabels(pv.index)
    step = 2
    ax.set_xticks(range(0, len(pv.columns), step))
    ax.set_xticklabels([c.strftime("%b %d") for c in pv.columns[::step]], rotation=45, ha="right")
    ax.set_title(f"Macro sentiment matrix: topic x week (mean {method_label()} score)")
    fig.colorbar(im, ax=ax, label="sentiment")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    import store
    OUT.mkdir(parents=True, exist_ok=True)
    ev, nw = store.events_df(), store.news_df()
    sentiment_timeline(ev, nw, OUT / "sentiment_timeline.png")
    sentiment_matrix(nw, OUT / "sentiment_matrix.png")
    print("charts ->", OUT)
if __name__ == "__main__":
    main()