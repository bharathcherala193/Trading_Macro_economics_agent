"""Step 5: walk-forward validation + strategy selection + risk metrics.

For each test week w (expanding window):
  train rows = earlier weeks whose labels were FULLY REALISED before w
               (label_known_ts <= w)  -> this is the purge/embargo
  model      = gradient boosting on the pooled (permutation x week) table
  decision   = pick the permutation with the highest predicted P&L
  score      = what that permutation ACTUALLY earned that week
Baselines it must beat to mean anything:
  trailing_best = permutation with the best realised P&L so far
  equal_weight  = average of all permutations (= a random pick in expectation)
"""
import argparse
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silences a harmless Windows wmic warning
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import config
import features as F

ANN = 52  
EVENT_PREFIXES = ("n_", "days_since_", "days_to_", "last_", "surprise_idx")

def is_event_feature(name: str) -> bool:
    return name.startswith(EVENT_PREFIXES)

def metrics(r: pd.Series) -> dict:
    r = pd.Series(r).astype(float)
    if len(r) < 2:
        return {}
    sd = r.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    eq = (1 + r).cumprod()
    return {"weeks": len(r),
            "total_return": eq.iloc[-1] - 1,
            "mean_weekly": r.mean(),
            "sharpe": r.mean() / sd * np.sqrt(ANN) if sd > 0 else np.nan,
            "sortino": r.mean() / downside * np.sqrt(ANN) if downside > 0 else np.nan,
            "max_drawdown": (eq / eq.cummax() - 1).min(),
            "hit_rate": (r > 0).mean()}

def rank_ic(pred, actual) -> float:
    if np.std(actual) == 0 or np.std(pred) == 0:
        return np.nan
    return float(np.corrcoef(pd.Series(pred).rank(), pd.Series(actual).rank())[0, 1])

def walk_forward(tbl: pd.DataFrame, min_train_weeks: int = 4, seed: int = 0,
                 use_events: bool = True) -> pd.DataFrame:
    cols = [c for c in F.feature_cols(tbl) if use_events or not is_event_feature(c)]
    weeks = sorted(tbl.week.unique())
    out = []
    for k in range(min_train_weeks, len(weeks)):
        w = weeks[k]
        train = tbl[(tbl.week < w) & (tbl.label_known_ts <= w)]       # purged
        test = tbl[tbl.week == w]
        if len(train) < 100 or train.week.nunique() < 3:
            continue
        
        usable = [c for c in cols if train[c].nunique(dropna=True) > 1]
        if not usable:
            pred = np.full(len(test), train.y.mean())
        else:
            model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=120,
                                                  min_samples_leaf=20, l2_regularization=1.0,
                                                  random_state=seed)
            model.fit(train[usable], train.y)
            pred = model.predict(test[usable])
        pick = int(np.argmax(pred))
        best_so_far = train.groupby("perm_id").y.sum().idxmax()
        out.append({"week": w,
                    "model": test.y.iloc[pick],
                    "trailing_best": test.loc[test.perm_id == best_so_far, "y"].iloc[0],
                    "equal_weight": test.y.mean(),
                    "oracle": test.y.max(),
                    "picked": test.perm_id.iloc[pick],
                    "rank_ic": rank_ic(pred, test.y.to_numpy()),
                    "n_train_rows": len(train)})
    return pd.DataFrame(out)

def report(res: pd.DataFrame) -> pd.DataFrame:
    rows = {name: metrics(res[name]) for name in ("model", "trailing_best", "equal_weight")}
    m = pd.DataFrame(rows).T
    print(m.round(4).to_string())
    print(f"\nmean weekly rank-IC (model vs realised, across permutations): {res.rank_ic.mean():.3f}")
    print(f"oracle (perfect foresight, unattainable upper bound) mean weekly: {res.oracle.mean():.4f}")
    if len(res) < 12:
        print(f"\nWARNING: only {len(res)} test weeks. Sharpe/Sortino on this few observations are "
              f"statistically meaningless; treat as a pipeline check, not a result.")
    return m

def main():
    import store
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", help="last date to include (default: last event date)")
    ap.add_argument("--min-train-weeks", type=int, default=4)
    ap.add_argument("--no-events", action="store_true",
                    help="drop calendar-event features; uses the FULL price/news period "
                         "(events currently stop earlier). Also serves as the ablation baseline.")
    args = ap.parse_args()

    ev, nw = store.events_df(), store.news_df()
    trades = pd.read_csv(config.TRADES_CSV)
    prices = pd.read_csv(config.PRICES_CSV, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index, utc=True)

    last_event = pd.to_datetime(ev.ts_utc, utc=True).max()
    if args.no_events:
        through = args.through            # None => all weeks the trades cover
        print(f"events cover up to {last_event:%Y-%m-%d}, but --no-events ignores them; "
              f"modelling through {through or 'end of data'}")
    else:
        through = args.through or last_event.strftime("%Y-%m-%d")
        print(f"events cover up to {last_event:%Y-%m-%d}; modelling through {through}")

    tbl = F.build_table(ev, nw, trades, prices, through=through)
    used = [c for c in F.feature_cols(tbl) if not (args.no_events and is_event_feature(c))]
    print(f"{tbl.week.nunique()} weeks x {tbl.perm_id.nunique()} permutations = {len(tbl)} rows, "
          f"{len(used)} features{' (events excluded)' if args.no_events else ''}\n")
    res = walk_forward(tbl, args.min_train_weeks, use_events=not args.no_events)
    if res.empty:
        raise SystemExit("Not enough weeks to run walk-forward. Re-scrape events or lower --min-train-weeks.")
    out = config.ROOT / "data" / ("walkforward_no_events.csv" if args.no_events else "walkforward_results.csv")
    res.to_csv(out, index=False)
    report(res)
    print(f"\nweekly results -> {out.name}")

if __name__ == "__main__":
    main()