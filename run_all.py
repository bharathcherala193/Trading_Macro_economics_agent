"""Run the whole pipeline in order. Stops at the first failure.

  python run_all.py              ingest live (asks before spending API credits), then everything else
  python run_all.py --yes        same, without the cost prompts
  python run_all.py --skip-ingest   reuse data already scraped/cached; rebuild the rest
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="skip cost confirmation prompts")
    ap.add_argument("--skip-ingest", action="store_true", help="don't scrape; use cached data")
    a = ap.parse_args()
    y = ["--yes"] if a.yes else []

    steps = []
    if not a.skip_ingest:
        steps += [("1/8 macro events (Apify)", "ingest_apify.py", y),
                  ("2/8 macro news (Tavily)", "ingest_news.py", y)]
    steps += [("3/8 trade logs", "trades.py", []),
              ("4/8 walk-forward, with events", "walkforward.py", []),
              ("5/8 walk-forward, without events", "walkforward.py", ["--no-events"]),
              ("6/8 sentiment charts", "charts.py", []),
              ("7/8 macro outlook summary", "macro_summary.py", []),
              ("8/8 evaluation report", "report.py", [])]
    for label, script, args in steps:
        print(f"\n=== {label} ===", flush=True)
        r = subprocess.run([sys.executable, str(HERE / script), *args], cwd=HERE)
        if r.returncode != 0:
            raise SystemExit(f"Step failed: {script} (exit {r.returncode})")
    print("\nDone. See outputs/ and EVALUATION_REPORT.md")

if __name__ == "__main__":
    main()