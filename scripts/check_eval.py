"""Print the P1 verification blocks from the evaluation reports.

Run with:  uv run python scripts/check_eval.py
"""
import json
from pathlib import Path

REPORT_DIR = Path("reports/rag_evaluation")
RESULTS = REPORT_DIR / "evaluation_results.json"
SUMMARY = REPORT_DIR / "evaluation_summary.json"

results = json.loads(RESULTS.read_text(encoding="utf-8"))
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))


def block(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


block("A. Latency per method (mean_latency_ms)")
aggregate = results.get("aggregate", {})
if not aggregate:
    print("MISSING: no 'aggregate' block in evaluation_results.json")
else:
    for method, metrics in aggregate.items():
        value = metrics.get("mean_latency_ms", "MISSING")
        print(f"  {method:30s} mean_latency_ms={value}")

block("B. Reproducibility block")
repro = results.get("reproducibility")
if not repro:
    print("MISSING: no 'reproducibility' block in evaluation_results.json")
else:
    for key, value in repro.items():
        if isinstance(value, str):
            print(f"  {key:20s} len={len(value):3d}  {value[:12]}...")
        else:
            print(f"  {key:20s} {value!r}")

block("C. Summary vs results consistency")
summary_agg = summary.get("aggregate_metrics") or summary.get("aggregate")
if not summary_agg:
    print("MISSING: no aggregate block in evaluation_summary.json")
else:
    try:
        for method in summary_agg:
            for key in ("precision@5", "recall@5", "mrr", "ndcg@5"):
                left = summary_agg[method][key]
                right = aggregate[method][key]
                assert abs(left - right) < 1e-9, (
                    method, key, left, right
                )
        print(f"  OK: summary and results agree on {len(summary_agg)} methods")
    except KeyError as exc:
        print(f"  FAIL: missing metric in one file: {exc}")
    except AssertionError as exc:
        print(f"  FAIL: mismatch {exc}")