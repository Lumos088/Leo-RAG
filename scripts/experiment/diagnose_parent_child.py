from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(r"E:\RAG")
LABELS = ROOT / "data/stage2/sufficiency/required_facts.jsonl"
EVIDENCE = ROOT / "data/stage2/evidence/evidence_units.jsonl"
LEGACY_SPANS = ROOT / "data/stage2/evidence/legacy_chunk_spans.jsonl"
QRELS = ROOT / "qrels.csv"
BASELINE_RUN = ROOT / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run"
PC_DIR = ROOT / "experiments/stage2/parent_child/minilm_c200_o50_v1_eval60"
PC_CHILD_RUN = PC_DIR / "parent_child_children.run"
PC_PARENT_RUN = PC_DIR / "parent_child_parents.run"
PC_DATA = ROOT / "data/stage2/parent_child/minilm_c200_o50_v1"
OUTPUT_JSON = PC_DIR / "failure_analysis.json"
OUTPUT_MD = PC_DIR / "failure_analysis.md"


def jsonl_by(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {
        str(item[key]): item
        for item in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def load_excluded_qids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    qids = payload.get("qids", payload) if isinstance(payload, dict) else payload
    if not isinstance(qids, list):
        raise ValueError("exclude-qids must be a JSON list or an object containing qids")
    return {str(qid) for qid in qids}


def load_json_arrays(directory: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            rows[str(item[key])] = item
    return rows


def load_run(path: Path) -> dict[str, list[tuple[str, int, float]]]:
    output: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        qid = parts[0]
        doc_id = " ".join(parts[2:-3])
        rank, score = parts[-3], parts[-2]
        output[qid].append((doc_id, int(rank), float(score)))
    for rows in output.values():
        rows.sort(key=lambda row: row[1])
    return dict(output)


def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for encoding in ("utf-8", "gb18030"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            pass
    raise ValueError(f"无法识别编码: {path}")


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    judged: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding=detect_encoding(path), newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"qid", "doc_id", "rel"}.issubset(reader.fieldnames):
            raise ValueError(f"qrels 必须包含 qid、doc_id、rel 列: {path}")
        for row in reader:
            qid = (row.get("qid") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            rel = (row.get("rel") or "").strip()
            if not qid or not doc_id or not rel:
                continue
            try:
                relevance = int(float(rel))
            except ValueError:
                continue
            judged[qid][doc_id] = max(relevance, judged[qid].get(doc_id, relevance))
    return dict(judged)


def span(item: Mapping[str, Any]) -> tuple[str, int, int] | None:
    source_id = str(item.get("source_id") or "")
    start = item.get("source_char_start", item.get("char_start"))
    end = item.get("source_char_end", item.get("char_end"))
    if not source_id or start is None or end is None:
        return None
    return source_id, int(start), int(end)


def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def evidence_match(
    candidates: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    relaxed: bool,
) -> set[str]:
    matched: set[str] = set()
    by_source: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for evidence_id, item in evidence.items():
        by_source[str(item["source_id"])].append((evidence_id, item))
    for candidate in candidates:
        candidate_span = span(candidate)
        if not candidate_span:
            continue
        source_id, start, end = candidate_span
        candidate_length = max(1, end - start)
        for evidence_id, item in by_source.get(source_id, []):
            evidence_start, evidence_end = int(item["char_start"]), int(item["char_end"])
            intersection = overlap((start, end), (evidence_start, evidence_end))
            if not intersection:
                continue
            evidence_length = max(1, evidence_end - evidence_start)
            if relaxed or intersection / evidence_length >= 0.50 or intersection / candidate_length >= 0.80:
                matched.add(evidence_id)
    return matched


def coverage(candidates: Sequence[Mapping[str, Any]], evidence_items: Sequence[Mapping[str, Any]]) -> float:
    if not evidence_items:
        return 0.0
    spans = [value for item in candidates if (value := span(item))]
    values: list[float] = []
    for item in evidence_items:
        source_id = str(item["source_id"])
        start, end = int(item["char_start"]), int(item["char_end"])
        intervals = sorted(
            (max(start, left), min(end, right))
            for candidate_source, left, right in spans
            if candidate_source == source_id and max(start, left) < min(end, right)
        )
        covered = 0
        cursor = start
        for left, right in intervals:
            if right <= cursor:
                continue
            covered += right - max(cursor, left)
            cursor = max(cursor, right)
        values.append(min(1.0, covered / max(1, end - start)))
    return sum(values) / len(values)


def fact_recall(label: Mapping[str, Any], matched_ids: set[str]) -> float:
    facts = label.get("required_facts", [])
    if not facts:
        return 0.0
    covered = sum(bool(set(item["supporting_evidence_ids"]) & matched_ids) for item in facts)
    return covered / len(facts)


def parent_values(rows: Sequence[tuple[str, int, float]], relevant: set[str], at: int = 10) -> tuple[float, int | None, float]:
    ids = [doc_id for doc_id, _rank, _score in rows]
    recall = len(set(ids[:at]) & relevant) / len(relevant) if relevant else 0.0
    first = next((rank for rank, doc_id in enumerate(ids[:at], 1) if doc_id in relevant), None)
    return recall, first, (1.0 / first if first else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--legacy-spans", type=Path, default=LEGACY_SPANS)
    parser.add_argument("--qrels", type=Path, default=QRELS)
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    parser.add_argument("--pc-dir", type=Path, default=PC_DIR)
    parser.add_argument("--pc-data", type=Path, default=PC_DATA)
    parser.add_argument("--exclude-qids", type=Path)
    args = parser.parse_args()
    output_json = args.pc_dir / "failure_analysis.json"
    output_md = args.pc_dir / "failure_analysis.md"

    labels = jsonl_by(args.labels, "qid")
    labels = {qid: item for qid, item in labels.items() if item.get("evaluation_status", "eligible") == "eligible"}
    excluded_qids = load_excluded_qids(args.exclude_qids)
    labels = {qid: item for qid, item in labels.items() if qid not in excluded_qids}
    evidence = jsonl_by(args.evidence, "evidence_id")
    legacy = jsonl_by(args.legacy_spans, "chunk_id")
    children = load_json_arrays(args.pc_data / "children", "child_id")
    qrels = load_qrels(args.qrels)
    baseline_run = load_run(args.baseline_run)
    pc_child_run = load_run(args.pc_dir / "parent_child_children.run")
    pc_parent_run = load_run(args.pc_dir / "parent_child_parents.run")

    per_query: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for qid in sorted(labels):
        label = labels[qid]
        supported_ids = set().union(*(set(item["supporting_evidence_ids"]) for item in label["required_facts"]))
        relevant_evidence = {item_id: evidence[item_id] for item_id in supported_ids if item_id in evidence}
        base_candidates = [legacy[doc_id] for doc_id, _rank, _score in baseline_run.get(qid, []) if doc_id in legacy]
        pc_candidates = [children[doc_id] for doc_id, _rank, _score in pc_child_run.get(qid, []) if doc_id in children]
        relevant_parents = {doc_id for doc_id, rel in qrels.get(qid, {}).items() if rel > 0}

        row: dict[str, Any] = {"qid": qid, "query": label["query"], "fact_count": len(label["required_facts"])}
        for cutoff in (5, 10, 50):
            base_strict = evidence_match(base_candidates[:cutoff], relevant_evidence, relaxed=False)
            base_relaxed = evidence_match(base_candidates[:cutoff], relevant_evidence, relaxed=True)
            pc_strict = evidence_match(pc_candidates[:cutoff], relevant_evidence, relaxed=False)
            pc_relaxed = evidence_match(pc_candidates[:cutoff], relevant_evidence, relaxed=True)
            row[f"baseline_fact_recall_strict@{cutoff}"] = fact_recall(label, base_strict)
            row[f"baseline_fact_recall_relaxed@{cutoff}"] = fact_recall(label, base_relaxed)
            row[f"pc_fact_recall_strict@{cutoff}"] = fact_recall(label, pc_strict)
            row[f"pc_fact_recall_relaxed@{cutoff}"] = fact_recall(label, pc_relaxed)
            row[f"baseline_evidence_coverage@{cutoff}"] = coverage(base_candidates[:cutoff], list(relevant_evidence.values()))
            row[f"pc_evidence_coverage@{cutoff}"] = coverage(pc_candidates[:cutoff], list(relevant_evidence.values()))

        base_recall10, base_first, base_mrr = parent_values(baseline_run.get(qid, []), relevant_parents)
        pc_recall10, pc_first, pc_mrr = parent_values(pc_parent_run.get(qid, []), relevant_parents)
        row.update({
            "baseline_parent_recall@10": base_recall10, "pc_parent_recall@10": pc_recall10,
            "baseline_first_relevant_parent": base_first, "pc_first_relevant_parent": pc_first,
            "baseline_parent_mrr@10": base_mrr, "pc_parent_mrr@10": pc_mrr,
        })

        b50 = row["baseline_fact_recall_strict@50"]
        p50 = row["pc_fact_recall_strict@50"]
        pr50 = row["pc_fact_recall_relaxed@50"]
        b10 = row["baseline_fact_recall_strict@10"]
        p10 = row["pc_fact_recall_strict@10"]
        if p50 == 0 and b50 > 0 and pr50 == 0:
            category = "child_retrieval_miss"
        elif pr50 - p50 >= 0.25:
            category = "span_threshold_penalty"
        elif p50 < b50 - 1e-9:
            category = "incomplete_child_coverage"
        elif pc_recall10 < base_recall10 - 1e-9 or pc_mrr < base_mrr - 1e-9:
            category = "parent_ranking_drop"
        elif p10 < b10 - 1e-9:
            category = "child_ranking_drop"
        elif p10 > b10 + 1e-9 or pc_recall10 > base_recall10 + 1e-9:
            category = "parent_child_improved"
        else:
            category = "stable"
        row["primary_diagnosis"] = category
        row["severity"] = round(
            (b10 - p10) + (b50 - p50) + (base_recall10 - pc_recall10) + (base_mrr - pc_mrr), 6
        )
        category_counts[category] += 1
        per_query.append(row)

    def mean(field: str) -> float:
        return sum(float(row[field]) for row in per_query) / len(per_query)

    aggregate: dict[str, Any] = {
        "queries": len(per_query),
        "category_counts": dict(category_counts.most_common()),
        "fact_metrics": {},
    }
    for cutoff in (5, 10, 50):
        aggregate["fact_metrics"][f"@{cutoff}"] = {
            "baseline_strict_mean_fact_recall": mean(f"baseline_fact_recall_strict@{cutoff}"),
            "parent_child_strict_mean_fact_recall": mean(f"pc_fact_recall_strict@{cutoff}"),
            "parent_child_relaxed_mean_fact_recall": mean(f"pc_fact_recall_relaxed@{cutoff}"),
            "baseline_all_facts_covered_rate": sum(row[f"baseline_fact_recall_strict@{cutoff}"] == 1 for row in per_query) / len(per_query),
            "parent_child_all_facts_covered_rate": sum(row[f"pc_fact_recall_strict@{cutoff}"] == 1 for row in per_query) / len(per_query),
            "parent_child_relaxed_all_facts_covered_rate": sum(row[f"pc_fact_recall_relaxed@{cutoff}"] == 1 for row in per_query) / len(per_query),
        }

    worst = sorted(per_query, key=lambda row: (-row["severity"], row["qid"]))[:15]
    report = {
        "schema_version": "parent-child-failure-analysis-v1",
        "match_rules": {
            "strict": "same source and either evidence coverage >= 0.50 or candidate purity >= 0.80",
            "relaxed": "same source and any positive character overlap",
        },
        "aggregate": aggregate,
        "worst_regressions": worst,
        "per_query": per_query,
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    names = {
        "child_retrieval_miss": "Child 未召回",
        "span_threshold_penalty": "跨度阈值惩罚",
        "incomplete_child_coverage": "Child 覆盖不足",
        "parent_ranking_drop": "Parent 排名下降",
        "child_ranking_drop": "Child 前排排序下降",
        "parent_child_improved": "Parent-Child 改善",
        "stable": "基本持平",
    }
    lines = [
        "# Parent-Child 逐题失败诊断", "",
        f"有效问题：{len(per_query)}", "", "## 主因分布", "",
    ]
    for key, count in category_counts.most_common():
        lines.append(f"- {names[key]}：{count} 题")
    lines.extend(["", "## 事实覆盖", "", "| 截断 | Baseline严格平均 | Parent-Child严格平均 | Parent-Child宽松平均 | Baseline完整率 | Parent-Child完整率 | Parent-Child宽松完整率 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for cutoff in (5, 10, 50):
        item = aggregate["fact_metrics"][f"@{cutoff}"]
        lines.append(
            f"| @{cutoff} | {item['baseline_strict_mean_fact_recall']:.4f} | {item['parent_child_strict_mean_fact_recall']:.4f} | "
            f"{item['parent_child_relaxed_mean_fact_recall']:.4f} | {item['baseline_all_facts_covered_rate']:.4f} | "
            f"{item['parent_child_all_facts_covered_rate']:.4f} | {item['parent_child_relaxed_all_facts_covered_rate']:.4f} |"
        )
    lines.extend(["", "## 回退最明显的15题", "", "| QID | 诊断 | Baseline事实@10 | PC事实@10 | PC宽松事实@10 | Baseline Parent@10 | PC Parent@10 |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in worst:
        lines.append(
            f"| {row['qid']} | {names[row['primary_diagnosis']]} | {row['baseline_fact_recall_strict@10']:.3f} | "
            f"{row['pc_fact_recall_strict@10']:.3f} | {row['pc_fact_recall_relaxed@10']:.3f} | "
            f"{row['baseline_parent_recall@10']:.3f} | {row['pc_parent_recall@10']:.3f} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_json), "markdown": str(output_md), **aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
