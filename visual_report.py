"""Step 9: ONE self-contained visual report (outputs/visual_report.html).

Why HTML: it opens in any browser with the charts embedded (no image files to attach
separately), and the browser can print it to PDF (Ctrl+P > Save as PDF).
Contents: macro outlook, sentiment charts, model-vs-baseline performance chart,
and the full evaluation report. Everything is read from files the pipeline already wrote.
"""
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown
import pandas as pd

import config

OUT = config.ROOT / "outputs"
RUNS = [("Calendar events + news + market features", "walkforward_results.csv"),
        ("News + market features only (no calendar events)", "walkforward_no_events.csv")]
SERIES = [("model", "ML model", "#1f4e79"),
          ("trailing_best", "Best permutation so far", "#e67e22"),
          ("equal_weight", "Average of all permutations", "#7f8c8d")]


def performance_chart(runs, path):
    """Cumulative return (top) and drawdown (bottom) of the model vs the two baselines.
    Same numbers as the report's Sharpe/Sortino/drawdown table, just visible."""
    fig, axes = plt.subplots(2, len(runs), figsize=(6.5 * len(runs), 6.5), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.6]})
    for j, (title, res) in enumerate(runs):
        a1, a2 = axes[0][j], axes[1][j]
        x = pd.to_datetime(res["week"], utc=True).dt.tz_convert(None)
        for col, label, color in SERIES:
            eq = (1 + res[col].astype(float)).cumprod()
            a1.plot(x, (eq - 1) * 100, label=label, color=color, lw=2.2 if col == "model" else 1.6)
            a2.plot(x, (eq / eq.cummax() - 1) * 100, color=color, lw=1.6)
        n = len(res)
        a1.set_title(title + ("" if n >= 12 else f"\n(only {n} test weeks: not meaningful)"), fontsize=10)
        a1.axhline(0, color="black", lw=0.6)
        a1.set_ylabel("cumulative return (%)")
        a2.set_ylabel("drawdown (%)")
        a1.legend(fontsize=8, loc="best")
        for a in (a1, a2):
            a.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def img(path, alt):
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{data}">'


def md_body(path):
    """Markdown -> HTML, dropping the file's own title line (we add section titles)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return markdown.markdown("\n".join(lines), extensions=["tables"])


CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.5}
h1{border-bottom:2px solid #1f4e79;padding-bottom:.2rem;margin-top:2.2rem} h2{margin-top:1.6rem}
img{max-width:100%;margin:.6rem 0} table{border-collapse:collapse;margin:.8rem 0;font-size:.9rem}
th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left} th{background:#f0f4f8}
blockquote{border-left:4px solid #e67e22;margin-left:0;padding-left:1rem;color:#555}
code{background:#f4f4f4;padding:0 .2rem} @media print{h1{page-break-before:always} h1:first-of-type{page-break-before:avoid}}
"""


def main():
    need = [OUT / "macro_outlook.md", OUT / "sentiment_timeline.png", OUT / "sentiment_matrix.png",
            config.ROOT / "EVALUATION_REPORT.md"]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {missing}. Run charts.py, macro_summary.py and report.py first.")
    runs = [(t, pd.read_csv(config.ROOT / "data" / f)) for t, f in RUNS if (config.ROOT / "data" / f).exists()]
    if not runs:
        raise SystemExit("No walk-forward results found. Run walkforward.py first.")
    perf = OUT / "performance.png"
    performance_chart(runs, perf)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Macro-Event Strategy Selection: Visual Report</title><style>{CSS}</style></head><body>
<h1>Macro-Event Strategy Selection: Visual Report</h1>
<p>Generated {pd.Timestamp.now(tz='UTC'):%d %b %Y}. All numbers and charts are produced by the pipeline.</p>
<h1>1. Macro outlook</h1>{md_body(OUT / 'macro_outlook.md')}
<h1>2. Macro sentiment over time</h1>
<p>News sentiment with CPI, FOMC and payrolls release dates marked, and the topic-by-week matrix.</p>
{img(OUT / 'sentiment_timeline.png', 'Sentiment timeline')}{img(OUT / 'sentiment_matrix.png', 'Sentiment matrix')}
<h1>3. Model performance</h1>
<p>Walk-forward, out-of-sample weeks only. Top: cumulative return. Bottom: drawdown.</p>
{img(perf, 'Model performance vs baselines')}
<h1>4. Evaluation report</h1>{md_body(config.ROOT / 'EVALUATION_REPORT.md')}
</body></html>"""
    out = OUT / "visual_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"visual report -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("To make a PDF: open it in a browser, Ctrl+P, Save as PDF.")


if __name__ == "__main__":
    main()