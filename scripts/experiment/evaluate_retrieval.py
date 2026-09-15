"""Evaluate TREC run files with a single, reproducible qrels parser.

The legacy evaluator parsed rows by splitting on commas, which breaks document
IDs containing commas. This module uses ``csv.DictReader`` and evaluates every
query that has at least one positive relevance judgement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def detect_text_encoding(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unable to decode {path}")


def load_qrels(path: Path) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    encoding = detect_text_encoding(path)
    judged: Dict[str, Dict[str, int]] = defaultdict(dict)
    row_counts: Dict[str, int] = defaultdict(int)
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"qid", "doc_id", "rel"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"qrels must contain columns {sorted(required)}")
        for row in reader:
            qid = (row.get("qid") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            rel_text = (row.get("rel") or "").strip()
            if not qid or not doc_id or not rel_text:
                continue
            try:
                rel = int(float(rel_text))
            except ValueError:
                continue
            judged[qid][doc_id] = max(rel, judged[qid].get(doc_id, rel))
            row_counts[qid] += 1
    return dict(judged), dict(row_counts)


def load_run(path: Path) -> Dict[str, List[Tuple[str, int, float]]]:
    run: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid = parts[0]
            doc_id = " ".join(parts[2:-3])
            try:
                rank = int(parts[-3])
                score = float(parts[-2])
            except ValueError as exc:
                raise ValueError(f"Invalid run row {path}:{line_number}") from exc
            # Legacy generation opened existing run files in append mode. A
            # new rank-1 row therefore starts a newer block for the same qid;
            # keep the most recent block instead of mixing repeated runs.
            if rank == 1 and run[qid]:
                run[qid] = []
            run[qid].append((doc_id, rank, score))
    for qid, rows in list(run.items()):
        deduplicated: Dict[str, Tuple[str, int, float]] = {}
        for row in rows:
            doc_id = row[0]
            if doc_id not in deduplicated or row[1] < deduplicated[doc_id][1]:
                deduplicated[doc_id] = row
        run[qid] = sorted(deduplicated.values(), key=lambda item: item[1])
    return dict(run)


def dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def evaluate_run(
    run: Mapping[str, Sequence[Tuple[str, int, float]]],
    qrels: Mapping[str, Mapping[str, int]],
) -> Dict[str, float | int]:
    evaluation_qids = sorted(
        qid for qid, judgements in qrels.items() if any(rel > 0 for rel in judgements.values())
    )
    if not evaluation_qids:
        raise ValueError("No query has a positive relevance judgement")

    recall_50 = 0.0
    mrr_10 = 0.0
    ndcg_10 = 0.0
    success_1 = 0.0
    success_5 = 0.0
    success_10 = 0.0

    for qid in evaluation_qids:
        judgements = qrels[qid]
        relevant = {doc_id for doc_id, rel in judgements.items() if rel > 0}
        ranked = list(run.get(qid, ()))

        top50 = {doc_id for doc_id, rank, _ in ranked if rank <= 50}
        recall_50 += len(top50 & relevant) / len(relevant)

        first_relevant_rank = next(
            (rank for doc_id, rank, _ in ranked if rank <= 10 and doc_id in relevant), None
        )
        if first_relevant_rank is not None:
            mrr_10 += 1.0 / first_relevant_rank

        top10_ids = [doc_id for doc_id, rank, _ in ranked if rank <= 10]
        gains = [judgements.get(doc_id, 0) for doc_id in top10_ids]
        gains.extend([0] * (10 - len(gains)))
        ideal = sorted(judgements.values(), reverse=True)[:10]
        ideal.extend([0] * (10 - len(ideal)))
        ideal_dcg = dcg(ideal)
        ndcg_10 += dcg(gains) / ideal_dcg if ideal_dcg else 0.0

        for cutoff, accumulator in ((1, "success_1"), (5, "success_5"), (10, "success_10")):
            hit = any(doc_id in relevant for doc_id, rank, _ in ranked if rank <= cutoff)
            if hit:
                if accumulator == "success_1":
                    success_1 += 1
                elif accumulator == "success_5":
                    success_5 += 1
                else:
                    success_10 += 1

    count = len(evaluation_qids)
    return {
        "evaluated_queries": count,
        "run_queries": len(run),
        "MRR@10": mrr_10 / count,
        "Success@1": success_1 / count,
        "Success@5": success_5 / count,
        "Success@10": success_10 / count,
        "Recall@50": recall_50 / count,
        "NDCG@10": ndcg_10 / count,
    }


def evaluate_directory(qrels_path: Path, runs_dir: Path) -> Dict[str, object]:
    qrels, row_counts = load_qrels(qrels_path)
    positive_qids = sorted(
        qid for qid, values in qrels.items() if any(rel > 0 for rel in values.values())
    )
    results: Dict[str, object] = {
        "qrels_path": str(qrels_path.resolve()),
        "qrels_encoding": detect_text_encoding(qrels_path),
        "judged_queries": len(qrels),
        "positive_queries": len(positive_qids),
        "positive_qids": positive_qids,
        "rows_by_query": row_counts,
        "systems": {},
    }
    for run_path in sorted(runs_dir.glob("*.run")):
        results["systems"][run_path.stem] = evaluate_run(load_run(run_path), qrels)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, default=PROJECT_ROOT / "qrels.csv")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate_directory(args.qrels, args.runs_dir)
    print(
        f"qrels: judged={report['judged_queries']} positive={report['positive_queries']} "
        f"encoding={report['qrels_encoding']}"
    )
    for system, metrics in report["systems"].items():
        print(
            f"{system:20s} MRR@10={metrics['MRR@10']:.4f} "
            f"NDCG@10={metrics['NDCG@10']:.4f} Recall@50={metrics['Recall@50']:.4f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
