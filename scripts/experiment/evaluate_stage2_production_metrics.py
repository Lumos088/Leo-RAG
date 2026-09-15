#!/usr/bin/env python3
"""Evaluate the deployed Parent-RRF + one baseline-rescue ranking."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(r"E:\RAG")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiment"))

from compare_parent_aggregations import context_metrics  # noqa: E402
from diagnose_parent_child import jsonl_by, load_json_arrays  # noqa: E402
from evaluate_retrieval import dcg, load_qrels, load_run  # noqa: E402


def final_ranking(
    rrf: dict[str, list[tuple[str, int, float]]],
    baseline: dict[str, list[tuple[str, int, float]]],
    primary_k: int = 5,
) -> dict[str, list[tuple[str, int, float]]]:
    output = {}
    for qid, rrf_rows in rrf.items():
        selected = [doc_id for doc_id, _rank, _score in rrf_rows[:primary_k]]
        for baseline_index, (doc_id, _rank, _score) in enumerate(baseline.get(qid, [])[:primary_k]):
            if doc_id not in selected:
                selected.insert(min(baseline_index, len(selected)), doc_id)
                break
        output[qid] = [(doc_id, rank, 1.0 / rank) for rank, doc_id in enumerate(selected, 1)]
    return output


def ranking_metrics(run, qrels, cutoffs=(1, 3, 5, 6, 10, 50)):
    qids = sorted(qid for qid, rels in qrels.items() if any(rel > 0 for rel in rels.values()))
    result = {"evaluated_queries": len(qids)}
    recalls = {k: 0.0 for k in cutoffs}
    successes = {k: 0.0 for k in cutoffs}
    mrr10 = 0.0
    ndcgs = {k: 0.0 for k in (3, 5, 6, 10)}
    for qid in qids:
        rels = qrels[qid]
        relevant = {doc_id for doc_id, rel in rels.items() if rel > 0}
        rows = list(run.get(qid, []))
        for cutoff in cutoffs:
            ids = {doc_id for doc_id, rank, _score in rows if rank <= cutoff}
            recalls[cutoff] += len(ids & relevant) / len(relevant)
            successes[cutoff] += float(bool(ids & relevant))
        first = next((rank for doc_id, rank, _score in rows if rank <= 10 and doc_id in relevant), None)
        mrr10 += 1.0 / first if first else 0.0
        for cutoff in ndcgs:
            top = [doc_id for doc_id, rank, _score in rows if rank <= cutoff]
            gains = [rels.get(doc_id, 0) for doc_id in top] + [0] * (cutoff - len(top))
            ideal = sorted(rels.values(), reverse=True)[:cutoff]
            ideal += [0] * (cutoff - len(ideal))
            denom = dcg(ideal)
            ndcgs[cutoff] += dcg(gains) / denom if denom else 0.0
    result["MRR@10"] = mrr10 / len(qids)
    for cutoff, value in ndcgs.items():
        result[f"NDCG@{cutoff}"] = value / len(qids)
    for cutoff in cutoffs:
        result[f"Recall@{cutoff}"] = recalls[cutoff] / len(qids)
        result[f"Success@{cutoff}"] = successes[cutoff] / len(qids)
    return result


def main():
    labels = {
        qid: row for qid, row in jsonl_by(ROOT / "data/stage2/sufficiency/required_facts.jsonl", "qid").items()
        if row.get("evaluation_status", "eligible") == "eligible"
    }
    qrels, _ = load_qrels(ROOT / "qrels.csv")
    qrels = {qid: qrels[qid] for qid in labels if qid in qrels and any(rel > 0 for rel in qrels[qid].values())}
    baseline = load_run(ROOT / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run")
    rrf = load_run(ROOT / "experiments/stage2/parent_child/minilm_c200_o50_parent_rrf20_eval60/parent_child_parents.run")
    production = final_ranking(rrf, baseline)
    evidence = jsonl_by(ROOT / "data/stage2/evidence/evidence_units.jsonl", "evidence_id")
    legacy = jsonl_by(ROOT / "data/stage2/evidence/legacy_chunk_spans.jsonl", "chunk_id")
    parents = load_json_arrays(ROOT / "data/stage2/parent_child/minilm_c200_o50_v1/parents", "parent_id")

    results = {
        "baseline": {"ranking": ranking_metrics(baseline, qrels)},
        "parent_rrf20": {"ranking": ranking_metrics(rrf, qrels)},
        "production_parent_rrf20_rescue1": {"ranking": ranking_metrics(production, qrels)},
    }
    for name, run, metadata in (
        ("baseline", baseline, legacy),
        ("parent_rrf20", rrf, parents),
        ("production_parent_rrf20_rescue1", production, parents),
    ):
        results[name]["context_fact@3"] = context_metrics(run, metadata, labels, evidence, 3)
        results[name]["context_fact@5"] = context_metrics(run, metadata, labels, evidence, 5)
        results[name]["context_fact@6"] = context_metrics(run, metadata, labels, evidence, 6)

    out = ROOT / "experiments/stage2/production_metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage 2 Production Ranking Metrics", "",
        "| Strategy | MRR@10 | NDCG@5 | NDCG@6 | NDCG@10 | Recall@5 | Recall@6 | Success@1 | Success@5 | Fact@5 | Full-fact@5 | Fact@6 | Full-fact@6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "parent_rrf20", "production_parent_rrf20_rescue1"):
        item = results[name]
        rank = item["ranking"]
        c5, c6 = item["context_fact@5"], item["context_fact@6"]
        lines.append(
            f"| {name} | {rank['MRR@10']:.4f} | {rank['NDCG@5']:.4f} | {rank['NDCG@6']:.4f} | {rank['NDCG@10']:.4f} | {rank['Recall@5']:.4f} | {rank['Recall@6']:.4f} | "
            f"{rank['Success@1']:.4f} | {rank['Success@5']:.4f} | {c5['mean_fact_recall']:.4f} | {c5['all_facts_covered_rate']:.4f} | "
            f"{c6['mean_fact_recall']:.4f} | {c6['all_facts_covered_rate']:.4f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
