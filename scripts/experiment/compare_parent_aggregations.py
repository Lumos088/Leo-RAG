from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diagnose_parent_child import evidence_match, fact_recall, jsonl_by, load_json_arrays, load_run


ROOT = Path(r"E:\RAG")
BASELINE_RUN = ROOT / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run"
LABELS = ROOT / "data/stage2/sufficiency/required_facts.jsonl"
EVIDENCE = ROOT / "data/stage2/evidence/evidence_units.jsonl"
LEGACY = ROOT / "data/stage2/evidence/legacy_chunk_spans.jsonl"
PARENT_DATA = ROOT / "data/stage2/parent_child/minilm_c200_o50_v1/parents"
VARIANTS = {
    "max": ROOT / "experiments/stage2/parent_child/minilm_c200_o50_v1_eval60",
    "top2_mean": ROOT / "experiments/stage2/parent_child/minilm_c200_o50_top2_mean_eval60",
    "max_rank_bonus": ROOT / "experiments/stage2/parent_child/minilm_c200_o50_max_rank_bonus_eval60",
    "parent_rerank20": ROOT / "experiments/stage2/parent_child/minilm_c200_o50_parent_rerank20_eval60",
    "parent_rrf20": ROOT / "experiments/stage2/parent_child/minilm_c200_o50_parent_rrf20_eval60",
}
OUTPUT_DIR = ROOT / "experiments/stage2/parent_child/aggregation_comparison_eval60"


def context_metrics(
    run: dict[str, list[tuple[str, int, float]]],
    metadata: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    cutoff: int,
) -> dict[str, float]:
    recalls: list[float] = []
    complete: list[bool] = []
    for qid, label in sorted(labels.items()):
        ids = set().union(*(set(item["supporting_evidence_ids"]) for item in label["required_facts"]))
        relevant = {item_id: evidence[item_id] for item_id in ids if item_id in evidence}
        candidates = [metadata[doc_id] for doc_id, _rank, _score in run.get(qid, [])[:cutoff] if doc_id in metadata]
        matched = evidence_match(candidates, relevant, relaxed=False)
        value = fact_recall(label, matched)
        recalls.append(value)
        complete.append(value == 1.0)
    return {
        "mean_fact_recall": sum(recalls) / len(recalls),
        "all_facts_covered_rate": sum(complete) / len(complete),
    }


def main() -> None:
    labels = {
        qid: item for qid, item in jsonl_by(LABELS, "qid").items()
        if item.get("evaluation_status", "eligible") == "eligible"
    }
    evidence = jsonl_by(EVIDENCE, "evidence_id")
    legacy = jsonl_by(LEGACY, "chunk_id")
    parents = load_json_arrays(PARENT_DATA, "parent_id")
    results: dict[str, Any] = {}

    baseline_run = load_run(BASELINE_RUN)
    results["baseline"] = {
        "context_fact@3": context_metrics(baseline_run, legacy, labels, evidence, 3),
        "context_fact@5": context_metrics(baseline_run, legacy, labels, evidence, 5),
    }
    for name, directory in VARIANTS.items():
        report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
        run = load_run(directory / "parent_child_parents.run")
        results[name] = {
            "parent_metrics": report["parent_child"]["parent"],
            "context_fact@3": context_metrics(run, parents, labels, evidence, 3),
            "context_fact@5": context_metrics(run, parents, labels, evidence, 5),
        }
    results["baseline"]["parent_metrics"] = json.loads(
        (VARIANTS["max"] / "comparison.json").read_text(encoding="utf-8")
    )["baseline"]["parent"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "comparison.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Parent 聚合策略对比", "",
        "| 策略 | Parent Recall@5 | Parent MRR@10 | Parent NDCG@10 | Top-3事实覆盖 | Top-3完整率 | Top-5事实覆盖 | Top-5完整率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "max", "top2_mean", "max_rank_bonus", "parent_rerank20", "parent_rrf20"):
        item = results[name]
        parent = item["parent_metrics"]
        c3, c5 = item["context_fact@3"], item["context_fact@5"]
        lines.append(
            f"| {name} | {parent['ParentRecall@5']:.4f} | {parent['ParentMRR@10']:.4f} | {parent['ParentNDCG@10']:.4f} | "
            f"{c3['mean_fact_recall']:.4f} | {c3['all_facts_covered_rate']:.4f} | "
            f"{c5['mean_fact_recall']:.4f} | {c5['all_facts_covered_rate']:.4f} |"
        )
    (OUTPUT_DIR / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
