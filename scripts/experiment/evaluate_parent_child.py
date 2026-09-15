"""Compare the Stage-1 retriever with an experimental Parent-Child variant.

The comparison deliberately reports two levels:

* evidence retrieval: source-span coverage using the shared Stage-2 evidence set;
* parent ranking: legacy qrels, possible because v1 parents preserve legacy IDs.

This script does not change the production retrieval configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment.evaluate_retrieval import detect_text_encoding, load_qrels, load_run
from scripts.experiment.run_stage1_retrieval import Stage1BM25, reciprocal_rank_fusion
from scripts.pipeline.parent_child_retrieval import ParentChildRetriever


def load_topics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding=detect_text_encoding(path), newline="") as handle:
        return [
            {"qid": row["qid"].strip(), "query": row["query"].strip()}
            for row in csv.DictReader(handle)
            if (row.get("qid") or "").strip() and (row.get("query") or "").strip()
        ]


def load_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record[key])] = record
    return records


def load_evaluation_scope(path: Path) -> tuple[set[str], list[str]]:
    labels = load_jsonl(path, "qid")
    excluded = sorted(
        qid for qid, label in labels.items()
        if label.get("evaluation_status", "eligible") == "excluded"
    )
    return set(labels) - set(excluded), excluded


def load_excluded_qids(path: Path | None) -> set[str]:
    """Load qids reserved for a sealed holdout evaluation."""
    if path is None:
        return set()
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    qids = payload.get("qids", payload) if isinstance(payload, dict) else payload
    if not isinstance(qids, list):
        raise ValueError("exclude-qids must be a JSON list or an object containing qids")
    return {str(qid) for qid in qids}


def load_evidence_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load evidence qrels whose identifier column is ``evidence_id``."""
    judged: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding=detect_text_encoding(path), newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"qid", "evidence_id", "rel"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"evidence qrels must contain columns {sorted(required)}")
        for row in reader:
            qid = (row.get("qid") or "").strip()
            evidence_id = (row.get("evidence_id") or "").strip()
            rel_text = (row.get("rel") or "").strip()
            if not qid or not evidence_id or not rel_text:
                continue
            try:
                rel = int(float(rel_text))
            except ValueError:
                continue
            judged[qid][evidence_id] = max(rel, judged[qid].get(evidence_id, rel))
    return dict(judged)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def summarize_latency(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_seconds": round(statistics.fmean(values), 4) if values else 0.0,
        "median_seconds": round(statistics.median(values), 4) if values else 0.0,
        "p95_seconds": round(percentile(values, 0.95), 4),
        "total_seconds": round(sum(values), 3),
    }


def candidate_span(candidate: Mapping[str, Any]) -> tuple[str, int, int] | None:
    source_id = str(candidate.get("source_id") or "")
    start = candidate.get("source_char_start", candidate.get("char_start"))
    end = candidate.get("source_char_end", candidate.get("char_end"))
    if not source_id or start is None or end is None:
        return None
    return source_id, int(start), int(end)


def overlap_length(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def matched_evidence_ids(
    candidates: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    matched: set[str] = set()
    by_source: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for evidence_id, unit in evidence.items():
        by_source[str(unit["source_id"])].append((evidence_id, unit))
    for candidate in candidates:
        span = candidate_span(candidate)
        if span is None:
            continue
        source_id, start, end = span
        candidate_length = max(1, end - start)
        for evidence_id, unit in by_source.get(source_id, ()):
            evidence_start, evidence_end = int(unit["char_start"]), int(unit["char_end"])
            intersection = overlap_length((start, end), (evidence_start, evidence_end))
            if not intersection:
                continue
            evidence_length = max(1, evidence_end - evidence_start)
            if intersection / evidence_length >= 0.50 or intersection / candidate_length >= 0.80:
                matched.add(evidence_id)
    return matched


def evidence_coverage(
    candidates: Sequence[Mapping[str, Any]],
    positive_units: Sequence[Mapping[str, Any]],
) -> float:
    if not positive_units:
        return 0.0
    candidate_spans = [span for candidate in candidates if (span := candidate_span(candidate))]
    coverages: list[float] = []
    for unit in positive_units:
        source_id = str(unit["source_id"])
        start, end = int(unit["char_start"]), int(unit["char_end"])
        intervals: list[tuple[int, int]] = []
        for candidate_source, candidate_start, candidate_end in candidate_spans:
            if candidate_source != source_id:
                continue
            left, right = max(start, candidate_start), min(end, candidate_end)
            if left < right:
                intervals.append((left, right))
        covered = 0
        cursor = start
        for left, right in sorted(intervals):
            if right <= cursor:
                continue
            covered += right - max(cursor, left)
            cursor = max(cursor, right)
        coverages.append(min(1.0, covered / max(1, end - start)))
    return statistics.fmean(coverages)


def evidence_metrics(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence_qrels: Mapping[str, Mapping[str, int]],
    evidence: Mapping[str, Mapping[str, Any]],
    cutoffs: Iterable[int] = (5, 10, 50),
) -> dict[str, float]:
    qids = sorted(qid for qid, rels in evidence_qrels.items() if any(rel > 0 for rel in rels.values()))
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        recalls: list[float] = []
        coverages: list[float] = []
        for qid in qids:
            positive_ids = {evidence_id for evidence_id, rel in evidence_qrels[qid].items() if rel > 0}
            positive_units = [evidence[evidence_id] for evidence_id in positive_ids]
            candidates = list(results.get(qid, ()))[:cutoff]
            matched = matched_evidence_ids(candidates, {key: evidence[key] for key in positive_ids})
            recalls.append(len(matched) / len(positive_ids))
            coverages.append(evidence_coverage(candidates, positive_units))
        metrics[f"EvidenceRecall@{cutoff}"] = statistics.fmean(recalls)
        metrics[f"EvidenceCoverage@{cutoff}"] = statistics.fmean(coverages)
    return metrics


def parent_metrics(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, float | int]:
    qids = sorted(qid for qid, rels in qrels.items() if any(rel > 0 for rel in rels.values()))
    sums = defaultdict(float)
    for qid in qids:
        ranked = list(results.get(qid, ()))
        judgements = qrels[qid]
        relevant = {doc_id for doc_id, rel in judgements.items() if rel > 0}
        ids = [str(item["id"]) for item in ranked]
        for cutoff in (3, 5, 8, 10, 50):
            sums[f"ParentRecall@{cutoff}"] += len(set(ids[:cutoff]) & relevant) / len(relevant)
        first = next((rank for rank, doc_id in enumerate(ids[:10], 1) if doc_id in relevant), None)
        sums["ParentMRR@10"] += 1.0 / first if first else 0.0
        gains = [judgements.get(doc_id, 0) for doc_id in ids[:10]]
        gains.extend([0] * (10 - len(gains)))
        ideal = sorted(judgements.values(), reverse=True)[:10]
        ideal.extend([0] * (10 - len(ideal)))
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
        idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
        sums["ParentNDCG@10"] += dcg / idcg if idcg else 0.0
    return {"evaluated_queries": len(qids), **{key: value / len(qids) for key, value in sums.items()}}


def baseline_results(
    run_path: Path,
    spans: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    run = load_run(run_path)
    results: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in run.items():
        results[qid] = [
            {"id": doc_id, **spans[doc_id], "score": score}
            for doc_id, _rank, score in rows
            if doc_id in spans
        ]
    return results


def write_run(path: Path, results: Mapping[str, Sequence[Mapping[str, Any]]], tag: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for qid, ranked in results.items():
            for rank, item in enumerate(ranked, 1):
                score = float(item.get("parent_score", item.get("rerank_score", item.get("score", 0.0))))
                handle.write(f"{qid} Q0 {item['id']} {rank} {score:.8f} {tag}\n")
    temporary.replace(path)


def run_parent_child(
    topics: Sequence[Mapping[str, str]],
    config_path: Path,
    device: str | None,
    parent_output_k: int = 50,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[float], float]:
    loaded_at = time.perf_counter()
    engine = ParentChildRetriever(config_path=config_path, device=device, load_reranker=True)
    load_seconds = time.perf_counter() - loaded_at
    children: dict[str, list[dict[str, Any]]] = {}
    parents: dict[str, list[dict[str, Any]]] = {}
    latencies: list[float] = []
    for number, topic in enumerate(topics, 1):
        started = time.perf_counter()
        result = engine.search(
            topic["query"], parent_top_k=parent_output_k, use_bm25=True, use_rerank=True
        )
        latencies.append(time.perf_counter() - started)
        children[topic["qid"]] = [dict(item) for item in result["ranked_children"]]
        parents[topic["qid"]] = [dict(item) for item in result["parents"]]
        if number == 1 or number % 10 == 0 or number == len(topics):
            print(f"[Parent-Child {number:02d}/{len(topics)}] {topic['qid']} {latencies[-1]:.3f}s")
    return children, parents, latencies, load_seconds


def diversity_metrics(
    children: Mapping[str, Sequence[Mapping[str, Any]]],
    parents: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    duplicate_rates: list[float] = []
    unique_parents: list[float] = []
    source_diversity: list[float] = []
    for qid, ranked_children in children.items():
        top_children = list(ranked_children)[:50]
        unique = len({str(item.get("parent_id")) for item in top_children})
        duplicate_rates.append(1.0 - unique / max(1, len(top_children)))
        unique_parents.append(float(unique))
        top_parents = list(parents.get(qid, ()))[:10]
        source_diversity.append(float(len({str(item.get("source_id")) for item in top_parents})))
    return {
        "ChildDuplicateParentRate@50": statistics.fmean(duplicate_rates),
        "UniqueParents@50Children": statistics.fmean(unique_parents),
        "UniqueSources@10Parents": statistics.fmean(source_diversity),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=Path, default=PROJECT_ROOT / "topics.csv")
    parser.add_argument("--qrels", type=Path, default=PROJECT_ROOT / "qrels.csv")
    parser.add_argument(
        "--evidence-dir", type=Path, default=PROJECT_ROOT / "data/stage2/evidence"
    )
    parser.add_argument(
        "--sufficiency-labels",
        type=Path,
        default=PROJECT_ROOT / "data/stage2/sufficiency/required_facts.jsonl",
        help="Only questions with evaluation_status=eligible are evaluated.",
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=PROJECT_ROOT / "experiments/stage1/runs/minilm-raw-rerank_rerank.run",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/parent_child.json"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parent-output-k", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/stage2/parent_child/minilm_c200_o50_v1",
    )
    parser.add_argument(
        "--exclude-qids",
        type=Path,
        help="JSON holdout manifest; these qids are omitted from candidate metrics.",
    )
    args = parser.parse_args()

    runtime_config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    topics = load_topics(args.topics.resolve())
    eligible_qids, excluded_qids = load_evaluation_scope(args.sufficiency_labels.resolve())
    sealed_qids = load_excluded_qids(args.exclude_qids)
    eligible_qids -= sealed_qids
    excluded_qids = sorted(set(excluded_qids) | sealed_qids)
    topics = [topic for topic in topics if topic["qid"] in eligible_qids]
    legacy_qrels, _ = load_qrels(args.qrels.resolve())
    legacy_qrels = {qid: rels for qid, rels in legacy_qrels.items() if qid in eligible_qids}
    evidence_qrels = load_evidence_qrels(
        (args.evidence_dir / "evidence_qrels.csv").resolve()
    )
    evidence_qrels = {qid: rels for qid, rels in evidence_qrels.items() if qid in eligible_qids}
    evidence = load_jsonl((args.evidence_dir / "evidence_units.jsonl").resolve(), "evidence_id")
    spans = load_jsonl((args.evidence_dir / "legacy_chunk_spans.jsonl").resolve(), "chunk_id")
    baseline = baseline_results(args.baseline_run.resolve(), spans)

    print(f"topics={len(topics)} excluded={len(excluded_qids)} baseline={args.baseline_run.resolve()}")
    pc_children, pc_parents, latencies, load_seconds = run_parent_child(
        topics, args.config.resolve(), args.device, args.parent_output_k
    )

    baseline_report = {
        "evidence": evidence_metrics(baseline, evidence_qrels, evidence),
        "parent": parent_metrics(baseline, legacy_qrels),
    }
    pc_report = {
        "evidence": evidence_metrics(pc_children, evidence_qrels, evidence),
        "parent": parent_metrics(pc_parents, legacy_qrels),
        "retrieval_shape": diversity_metrics(pc_children, pc_parents),
        "timing": {"load_seconds": round(load_seconds, 3), **summarize_latency(latencies)},
    }
    report = {
        "schema_version": "parent-child-evaluation-v1",
        "comparison_note": "Evidence metrics compare Stage-1 chunks with ranked Child spans. Parent metrics use legacy qrels because v1 Parent IDs are unchanged.",
        "evidence_match_rule": "same source AND (evidence coverage >= 0.50 OR candidate purity >= 0.80)",
        "topics": len(topics),
        "excluded_topics": excluded_qids,
        "parent_score_aggregation": runtime_config.get("parent_score_aggregation", "max"),
        "parent_rank_bonus_weight": runtime_config.get("parent_rank_bonus_weight", 0.05),
        "parent_rerank_enabled": bool(runtime_config.get("parent_rerank_enabled", False)),
        "parent_rerank_candidates": int(runtime_config.get("parent_rerank_candidates", 20)),
        "parent_rerank_mode": runtime_config.get("parent_rerank_mode", "replace"),
        "parent_rrf_k": int(runtime_config.get("parent_rrf_k", 20)),
        "parent_child_rank_weight": float(runtime_config.get("parent_child_rank_weight", 1.0)),
        "parent_bge_rank_weight": float(runtime_config.get("parent_bge_rank_weight", 1.0)),
        "parent_output_k": args.parent_output_k,
        "baseline": baseline_report,
        "parent_child": pc_report,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "comparison.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run(args.output_dir / "parent_child_children.run", pc_children, "parent_child_children")
    write_run(args.output_dir / "parent_child_parents.run", pc_parents, "parent_child_parents")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={output_path.resolve()}")


if __name__ == "__main__":
    main()
