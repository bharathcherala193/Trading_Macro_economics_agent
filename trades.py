"""Step 3: generate a trade log for every strategy permutation.

Why this exists: the assignment says "ingest historical trading logs" but gives
none. So we backtest a parameterised moving-average-crossover strategy on SPY
for each parameter combination. Each trade is one row. The model later predicts
how each permutation will perform in a future period.

Anti-look-ahead rules baked in (reviewers check these first):
  1. A signal is computed from bar t-1's CLOSE and acted on at bar t's OPEN.
  2. If a stop and a target are both touched inside one bar, we assume the
     STOP hit first (conservative; we can't know the intrabar order).
  3. If a bar gaps through the stop, we fill at the open, not the stop price.
  4. Costs are charged on both sides of every trade.
"""
import argparse
import itertools
import numpy as np
import pandas as pd
import config

def load_prices(refresh: bool = False) -> pd.DataFrame:
    if config.PRICES_CSV.exists() and not refresh:
        # Yahoo's hourly history differs between downloads (bar counts changed run to run),
        # which silently changes every trade and result. So the first download is frozen
        # and committed; re-download only on purpose with --refresh.
        df = pd.read_csv(config.PRICES_CSV, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "ts_utc"
        print(f"using frozen prices: {config.PRICES_CSV.name} (python trades.py --refresh to re-download)")
        return df
    import yfinance as yf
    df = yf.download(config.TRADE_SYMBOL, start=str(config.START_DATE), end=str(config.END_DATE),
                     interval=config.TRADE_INTERVAL, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):       # newer yfinance returns (field, ticker)
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = "ts_utc"
    config.PRICES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PRICES_CSV)   # cached so the pipeline is reproducible offline
    return df

def synthetic_prices(days: int = 150, seed: int = 7) -> pd.DataFrame:
    """Random-walk hourly bars for testing the machinery without internet."""
    rng = np.random.default_rng(seed)
    days_idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    ts = [d + pd.Timedelta(hours=h) for d in days_idx for h in range(14, 21)]  # ~US session in UTC
    ret = rng.normal(0.00002, 0.0025, len(ts))
    close = 450 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0004, len(ts)))
    spread = np.abs(rng.normal(0, 0.0015, len(ts)))
    df = pd.DataFrame({"open": open_, "close": close,
                       "high": np.maximum(open_, close) * (1 + spread),
                       "low": np.minimum(open_, close) * (1 - spread),
                       "volume": 1e6}, index=pd.DatetimeIndex(ts, tz="UTC", name="ts_utc"))
    return df


# ----------------------------------------------------------- backtest -------
def backtest(df: pd.DataFrame, fast: int, slow: int, stop_loss: float, take_profit: float,
             cost_bps: float = config.COST_BPS) -> list[dict]:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = df.index
    f = pd.Series(c).rolling(fast).mean().to_numpy()
    s = pd.Series(c).rolling(slow).mean().to_numpy()
    cost = 2 * cost_bps / 1e4
    trades, pos, entry, entry_i = [], 0, 0.0, 0

    def close_trade(i, price, reason):
        nonlocal pos
        gross = pos * (price / entry - 1)
        trades.append({"entry_ts": idx[entry_i], "exit_ts": idx[i], "side": pos,
                       "entry_px": entry, "exit_px": price, "ret": gross - cost, "exit_reason": reason})
        pos = 0

    for t in range(slow + 1, len(df)):
        # signal uses ONLY data up to the close of bar t-1
        sig = 0
        if f[t - 1] > s[t - 1] and f[t - 2] <= s[t - 2]:
            sig = 1
        elif f[t - 1] < s[t - 1] and f[t - 2] >= s[t - 2]:
            sig = -1

        if pos != 0 and sig == -pos:                 # opposite signal: exit at this open
            close_trade(t, o[t], "reverse")
        if pos == 0 and sig != 0:                    # enter at this open
            pos, entry, entry_i = sig, o[t], t

        if pos != 0:                                 # stop / target inside bar t
            stop_px = entry * (1 - sl_dir(pos) * stop_loss)
            tgt_px = entry * (1 + sl_dir(pos) * take_profit)
            if pos == 1:
                hit_stop, hit_tgt = l[t] <= stop_px, h[t] >= tgt_px
                gap_stop = o[t] <= stop_px and t > entry_i
            else:
                hit_stop, hit_tgt = h[t] >= stop_px, l[t] <= tgt_px
                gap_stop = o[t] >= stop_px and t > entry_i
            if hit_stop:                             # stop wins ties (conservative)
                close_trade(t, o[t] if gap_stop else stop_px, "stop")
            elif hit_tgt:
                close_trade(t, tgt_px, "target")

    if pos != 0:                                     # data ended while in a trade
        close_trade(len(df) - 1, c[-1], "end_of_data")
    return trades

def sl_dir(pos: int) -> int:
    return pos  # long: stop below / target above; short: mirrored

def run_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fast, slow, sl, tp in itertools.product(*config.PARAM_GRID.values()):
        if fast >= slow:
            continue
        pid = f"f{fast}_s{slow}_sl{sl}_tp{tp}"
        for tr in backtest(df, fast, slow, sl, tp):
            rows.append({"perm_id": pid, "fast": fast, "slow": slow, "stop_loss": sl,
                         "take_profit": tp, **tr})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="random-walk prices (offline test only)")
    args = ap.parse_args()
    df = synthetic_prices() if args.synthetic else load_prices()
    print(f"{len(df)} bars, {df.index.min():%Y-%m-%d} -> {df.index.max():%Y-%m-%d}")
    tr = run_grid(df)
    out = config.TRADES_CSV.with_name("trades_synthetic.csv") if args.synthetic else config.TRADES_CSV
    tr.to_csv(out, index=False)
    print(f"{len(tr)} trades across {tr.perm_id.nunique()} permutations -> {out.name}")
    print(tr.groupby("perm_id").ret.agg(["count", "mean"]).describe().loc[["min", "50%", "max"]].round(5).to_string())

if __name__ == "__main__":
    main()