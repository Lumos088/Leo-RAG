"""Aggregate Stage-1 model metrics without rerunning retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage1" / "runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage1" / "model_comparison.json",
    )
    parser.add_argument("--recall-retention", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.recall_retention <= 1:
        raise ValueError("recall-retention must be in (0, 1]")

    rows = []
    for path in sorted(args.metrics_dir.glob("*_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for system_name, metrics in payload["systems"].items():
            rows.append(
                {
                    "label": payload["label"],
                    "system": system_name,
                    "model": payload["embedding"]["model_name"],
                    "text_mode": payload.get("index", {}).get("text_mode", "unknown"),
                    "dimension": payload.get("index", {}).get("dimension"),
                    "index_bytes": payload.get("index", {}).get("index_bytes"),
                    "offline_encode_seconds": payload.get("index", {}).get("offline_encode_seconds"),
                    "MRR@10": metrics["MRR@10"],
                    "NDCG@10": metrics["NDCG@10"],
                    "Recall@50": metrics["Recall@50"],
                    "mean_query_seconds": payload["timing_seconds"]["query_mean"],
                }
            )
    if not rows:
        raise FileNotFoundError(f"No *_metrics.json files in {args.metrics_dir}")
    rows.sort(key=lambda row: (row["system"], -row["NDCG@10"], -row["Recall@50"]))
    hybrid_rows = [row for row in rows if row["system"] == "hybrid"]
    best_recall = max(row["Recall@50"] for row in hybrid_rows)
    recall_floor = args.recall_retention * best_recall
    eligible = [row for row in hybrid_rows if row["Recall@50"] >= recall_floor]
    selected = max(
        eligible,
        key=lambda row: (row["NDCG@10"], row["Recall@50"], -row["mean_query_seconds"]),
    )
    report = {
        "comparison_scope": "Queries with at least one positive qrels judgement",
        "ranking_note": "Quality is primary; latency is reported separately and is not folded into an arbitrary score.",
        "selection_rule": (
            f"Keep hybrid variants with Recall@50 >= {args.recall_retention:.0%} of the best "
            "observed Recall@50, then maximize NDCG@10."
        ),
        "recall_floor": recall_floor,
        "eligible_labels": [row["label"] for row in eligible],
        "recommended_label": selected["label"],
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'label':22s} {'system':8s} {'NDCG@10':>9s} {'Recall@50':>10s} {'sec/query':>10s} {'dim':>6s}")
    for row in rows:
        print(
            f"{row['label']:22s} {row['system']:8s} {row['NDCG@10']:9.4f} "
            f"{row['Recall@50']:10.4f} {row['mean_query_seconds']:10.4f} "
            f"{str(row['dimension'] or '-'):>6s}"
        )
    print(f"recommended={selected['label']} recall_floor={recall_floor:.4f}")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
