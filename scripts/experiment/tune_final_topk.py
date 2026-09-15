"""Choose the final number of passages sent to generation on a dev split.

Selection rule: retain configurations reaching at least 95% of the best dev
Recall among the tested K values, then choose the highest dev NDCG (smaller K
wins exact ties). The held-out split is reported only after the choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment.evaluate_retrieval import load_qrels, load_run


def split_qids(qids: Sequence[str], test_ratio: float, seed: str) -> tuple[list[str], list[str]]:
    ordered = sorted(
        qids,
        key=lambda qid: hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest(),
    )
    test_count = max(1, round(len(ordered) * test_ratio))
    return sorted(ordered[test_count:]), sorted(ordered[:test_count])


def metrics_at_k(
    qids: Sequence[str],
    run: Mapping[str, Sequence[tuple[str, int, float]]],
    qrels: Mapping[str, Mapping[str, int]],
    k: int,
) -> dict[str, float | int]:
    recall = precision = ndcg = success = 0.0
    for qid in qids:
        judgements = qrels[qid]
        relevant = {doc_id for doc_id, rel in judgements.items() if rel > 0}
        ranked_ids = [doc_id for doc_id, rank, _ in run.get(qid, ()) if rank <= k]
        hits = sum(doc_id in relevant for doc_id in ranked_ids)
        recall += hits / len(relevant)
        precision += hits / k
        success += float(hits > 0)
        gains = [judgements.get(doc_id, 0) for doc_id in ranked_ids]
        gains.extend([0] * (k - len(gains)))
        ideal = sorted(judgements.values(), reverse=True)[:k]
        ideal.extend([0] * (k - len(ideal)))
        actual_dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(gains))
        ideal_dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
        ndcg += actual_dcg / ideal_dcg if ideal_dcg else 0.0
    count = len(qids)
    return {
        "queries": count,
        f"Recall@{k}": recall / count,
        f"Precision@{k}": precision / count,
        f"NDCG@{k}": ndcg / count,
        f"Success@{k}": success / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, default=PROJECT_ROOT / "qrels.csv")
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 8, 10])
    parser.add_argument("--recall-retention", type=float, default=0.95)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", default="stage1-v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "stage1" / "topk.json")
    args = parser.parse_args()

    k_values = sorted(set(args.k_values))
    if not k_values or k_values[0] <= 0:
        raise ValueError("k-values must be positive")
    if not 0 < args.recall_retention <= 1:
        raise ValueError("recall-retention must be in (0, 1]")
    if not 0 < args.test_ratio < 1:
        raise ValueError("test-ratio must be in (0, 1)")

    qrels, _ = load_qrels(args.qrels.resolve())
    positive_qids = sorted(
        qid for qid, values in qrels.items() if any(rel > 0 for rel in values.values())
    )
    development_qids, test_qids = split_qids(positive_qids, args.test_ratio, args.seed)
    run = load_run(args.run.resolve())
    development = {str(k): metrics_at_k(development_qids, run, qrels, k) for k in k_values}

    maximum_recall = max(float(development[str(k)][f"Recall@{k}"]) for k in k_values)
    recall_floor = args.recall_retention * maximum_recall
    eligible = [
        k for k in k_values if float(development[str(k)][f"Recall@{k}"]) >= recall_floor
    ]
    selected_k = max(
        eligible,
        key=lambda k: (float(development[str(k)][f"NDCG@{k}"]), -k),
    )
    test = metrics_at_k(test_qids, run, qrels, selected_k)
    report = {
        "run": str(args.run.resolve()),
        "selection_rule": (
            "On development queries, retain K values with Recall@K >= "
            f"{args.recall_retention:.0%} of the best tested Recall@K; then maximize NDCG@K; "
            "prefer smaller K on exact ties."
        ),
        "k_values": k_values,
        "development_qids": development_qids,
        "test_qids": test_qids,
        "development": development,
        "recall_floor": recall_floor,
        "eligible_k": eligible,
        "selected_k": selected_k,
        "held_out_test": test,
        "warning": (
            "Only queries with at least one positive judgement are used. Precision is a lower-bound "
            "estimate when qrels are incompletely pooled."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for k in k_values:
        row = development[str(k)]
        print(
            f"K={k:2d} dev Recall={row[f'Recall@{k}']:.4f} "
            f"NDCG={row[f'NDCG@{k}']:.4f} Precision={row[f'Precision@{k}']:.4f}"
        )
    print(f"selected K={selected_k}; held-out={json.dumps(test, ensure_ascii=False)}")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
