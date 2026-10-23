"""Step 8 (assignment deliverable 3): evaluation report with Sharpe, Sortino, max drawdown.

Numbers are filled in from the walk-forward result files, and the interpretation
sentences are chosen from those numbers, so the report cannot say something the
results don't support. Output: EVALUATION_REPORT.md in the repo root.
"""
import json

import numpy as np
import pandas as pd

import config
import walkforward as W

LABELS = {"model": "ML model (picks a permutation each week)",
          "trailing_best": "Baseline: best permutation so far",
          "equal_weight": "Baseline: average of all permutations"}
RUNS = [("Calendar events + news + market features", "walkforward_results.csv"),
        ("News + market features only (no calendar events)", "walkforward_no_events.csv")]


def f(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.2f}"


def metrics_table(res: pd.DataFrame) -> str:
    out = ["| Strategy | Sharpe | Sortino | Max drawdown | Total return | Hit rate | Weeks |",
           "|---|---|---|---|---|---|---|"]
    for k, lab in LABELS.items():
        m = W.metrics(res[k])
        out.append(f"| {lab} | {f(m['sharpe'])} | {f(m['sortino'])} | {f(m['max_drawdown'], True)} | "
                   f"{f(m['total_return'], True)} | {f(m['hit_rate'], True)} | {m['weeks']} |")
    return "\n".join(out)


def interpret(title: str, res: pd.DataFrame) -> str:
    n = len(res)
    if n < 12:
        return (f"**{title}.** Only {n} out-of-sample weeks are available, far too few for Sharpe or Sortino "
                f"to mean anything. The figures above are shown for completeness and support no conclusion.")
    m = {k: W.metrics(res[k]) for k in LABELS}
    sm, se_, st = (m[k]["sharpe"] for k in ("model", "equal_weight", "trailing_best"))
    ic = res.rank_ic.mean()
    se = np.sqrt(W.ANN / n)  # approximate standard error of an annualised Sharpe over n weeks
    t = (f"**{title}.** Over {n} out-of-sample weeks the model's Sharpe was {f(sm)}, against {f(se_)} for the "
         f"average of all permutations and {f(st)} for the best-so-far rule. Mean weekly rank-IC (does the "
         f"model rank permutations in the right order?) was {ic:+.3f}. ")
    if np.isnan(sm) or np.isnan(se_) or ic <= 0.05 or sm <= se_:
        t += "The model did not show a reliable ability to pick better permutations than chance. "
    else:
        t += "Its rankings were weakly positive and its Sharpe was above the baseline's. "
    if not (np.isnan(sm) or np.isnan(se_)) and abs(sm - se_) < 2 * se:
        t += (f"With {n} weeks the approximate standard error of an annualised Sharpe is about "
              f"+/-{se:.1f}, so gaps of this size cannot be told apart from luck.")
    else:
        t += (f"The gap is larger than two approximate standard errors (+/-{se:.1f} each), "
              f"but {n} weeks is still a short sample.")
    return t


def facts() -> dict:
    import store
    ev, nw = store.events_df(), store.news_df()
    tr = pd.read_csv(config.TRADES_CSV)
    px = pd.read_csv(config.PRICES_CSV, index_col=0, parse_dates=True)
    cache = config.ROOT / "data" / "sentiment_cache.json"
    keys = list(json.loads(cache.read_text())) if cache.exists() else []
    d = lambda s: pd.to_datetime(s, utc=True)
    return {"n_events": len(ev), "ev0": d(ev.ts_utc).min(), "ev1": d(ev.ts_utc).max(),
            "n_news": len(nw), "nw0": d(nw.published_utc).min(), "nw1": d(nw.published_utc).max(),
            "n_bars": len(px), "px0": d(px.index).min(), "px1": d(px.index).max(),
            "n_trades": len(tr), "n_perms": tr.perm_id.nunique(),
            "finbert": any(k.startswith("finbert:") for k in keys),
            "has_nfp": bool(ev.name.astype(str).str.lower().str.contains("nonfarm").any())}


def build(fx: dict, runs: list) -> str:
    sent = ("FinBERT (ProsusAI/finbert), score = P(positive) - P(negative)" if fx["finbert"]
            else "a simple word list (positive minus negative term counts); crude, see limitations")
    gap = fx["ev1"] < fx["px1"] - pd.Timedelta(days=14)
    L = ["# Evaluation Report: Macro-Event Strategy Selection",
         f"*Generated {pd.Timestamp.now(tz='UTC'):%d %b %Y}. Numbers below are produced by the pipeline, not typed in.*",
         "\n## 1. What was built",
         "A pipeline that (1) scrapes a US macro calendar and recent macro news, (2) generates trade logs for 36 "
         "parameter permutations of a moving-average strategy on SPY, (3) builds weekly time-based, event, news and "
         "market features, (4) trains a model to predict each permutation's next-week P&L, and (5) selects the "
         "predicted-best permutation each week under walk-forward validation.",
         "\n## 2. Data",
         f"- **Events:** Apify actor `{config.APIFY_ACTOR_ID}` (Investing.com calendar), filtered to "
         f"{', '.join(config.ACTOR_FILTERS['countries'])}, importance {', '.join(config.ACTOR_FILTERS['importances'])}. "
         f"{fx['n_events']} events, {fx['ev0']:%d %b %Y} to {fx['ev1']:%d %b %Y}. Times stored in UTC and US Eastern.",
         f"- **News:** Tavily search, topic `{config.NEWS_TOPIC}`, {len(config.NEWS_QUERIES)} queries "
         f"({'; '.join(config.NEWS_QUERIES)}), {config.NEWS_WINDOW_DAYS}-day windows. {fx['n_news']} unique articles, "
         f"{fx['nw0']:%d %b %Y} to {fx['nw1']:%d %b %Y}.",
         f"- **Prices and trade logs:** no trade logs were provided, so they were generated by backtesting on "
         f"{config.TRADE_SYMBOL} {config.TRADE_INTERVAL} bars ({fx['n_bars']} bars, {fx['px0']:%d %b} to "
         f"{fx['px1']:%d %b %Y}), giving {fx['n_trades']} trades across {fx['n_perms']} permutations "
         f"(fast MA, slow MA, stop-loss, take-profit; {config.COST_BPS:g} bp cost per side).",
         f"- **Sentiment:** {sent}.",
         "\n## 3. Method",
         "- **Target:** total net return of the trades a permutation entered in a week (0 if it stayed flat).",
         "- **Features (known at the start of the week):** days since/until the next CPI, FOMC decision and jobs "
         "report; number of high-impact events scheduled that week; sign of the last CPI/jobs surprise and a 4-week "
         "surprise index; news sentiment over 7 and 28 days, its change, and article count; prior-week SPY return, "
         "volatility and range; each permutation's recent P&L and win rate; the permutation parameters.",
         "- **Model:** gradient boosting (scikit-learn HistGradientBoostingRegressor: depth 3, learning rate 0.05, "
         "120 iterations, min leaf 20, L2 1.0), one model on all permutation-weeks. No tuning on test data.",
         "- **Walk-forward:** expanding window, one test week at a time, at least 4 training weeks. A training row is "
         "used only if all its trades had exited before the test week began (purging), because trades overlap weeks.",
         "- **Selection and scoring:** each test week the model picks the permutation with the highest predicted P&L, "
         "and the score is what that permutation actually earned. Metrics use weekly returns, annualised with sqrt(52); "
         "max drawdown is taken on the compounded equity curve.",
         "\n## 4. Look-ahead controls",
         "- Strategy signals use the previous bar's close and trade at the next bar's open; if one bar touches both "
         "stop and target the stop is assumed to hit first; gaps fill at the open; costs are charged both ways.",
         "- Features use only information dated before the week starts, except the event *schedule* (what is due, "
         "days to the next release), which is published in advance. Actual release values are never used for the "
         "week they belong to.",
         "- Permutation history uses only trades that have already exited.",
         "\n## 5. Results"]
    for title, res in runs:
        note = " Sortino is n/a for a series with no losing week." if (res[list(LABELS)].min() >= 0).any() else ""
        L += [f"\n### {title}", metrics_table(res),
              f"\nMean weekly rank-IC: {res.rank_ic.mean():+.3f}.{note}",
              "\n" + interpret(title, res)]
    L += ["\n## 6. Limitations",
          f"- **Short sample.** Weekly walk-forward over about {max(len(r) for _, r in runs)} test weeks. The 36 "
          f"permutations trade the same prices, so the effective sample is far smaller than the row count.",
          "- **Self-generated trade logs** from one simple strategy on one instrument. Results say nothing about the "
          "company's own strategies; the pipeline accepts a real trade log with the same columns."]
    if gap:
        L.append(f"- **Event coverage.** Calendar events end {fx['ev1']:%d %b %Y}, earlier than prices and news "
                 f"({fx['px1']:%d %b %Y}). The event-feature run is limited to the covered weeks, and the second run "
                 f"excludes events so it can use the full period.")
    if not fx.get("has_nfp", True):
        L.append("- **No Nonfarm Payrolls events are in the event data,** so the jobs-report features are empty. "
                 "The calendar scrape was cut off by a row cap and needs re-running with the category filter.")
    L += ["- **Sentiment is approximate**, and the model's use of it is only as good as the scorer.",
          "- **Two feature sets** were evaluated (with and without calendar events); no other variants were tried.",
          "\n## 7. Reproduce",
          "`pip install -r requirements.txt`, copy `.env.example` to `.env` and add your own keys, then "
          "`python run_all.py`. See README."]
    return "\n".join(L)


def main():
    runs = []
    for title, fn in RUNS:
        p = config.ROOT / "data" / fn
        if p.exists():
            runs.append((title, pd.read_csv(p)))
    if not runs:
        raise SystemExit("No walk-forward results found. Run walkforward.py first.")
    md = build(facts(), runs)
    out = config.ROOT / "EVALUATION_REPORT.md"
    out.write_text(md, encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()