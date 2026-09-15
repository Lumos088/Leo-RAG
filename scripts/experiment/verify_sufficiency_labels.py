from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = PROJECT_DIR / "data" / "stage2" / "sufficiency" / "required_facts.jsonl"
DEFAULT_EVIDENCE = PROJECT_DIR / "data" / "stage2" / "evidence" / "evidence_units.jsonl"
DEFAULT_TOPICS = PROJECT_DIR / "topics.csv"
DEFAULT_REPORT = PROJECT_DIR / "data" / "stage2" / "sufficiency" / "verification_report.json"
DEFAULT_REVIEW = PROJECT_DIR / "data" / "stage2" / "sufficiency" / "review_queue.csv"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_topics(path: Path) -> dict[str, str]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return {row["qid"]: row["query"] for row in csv.DictReader(handle)}
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别 topics 编码: {path}")


def primary_course(label: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> str:
    courses = Counter(
        evidence_by_id[evidence_id].get("course", "Unknown")
        for evidence_id in label.get("candidate_evidence_ids", [])
        if evidence_id in evidence_by_id
    )
    return courses.most_common(1)[0][0] if courses else "Unknown"


def make_review_row(label: dict[str, Any], course: str, reason: str) -> dict[str, str]:
    return {
        "qid": label["qid"], "query": label.get("query", ""), "primary_course": course,
        "query_type": label.get("query_type", ""), "fact_count": str(len(label.get("required_facts", []))),
        "evidence_complete": str(bool(label.get("evidence_complete"))).lower(),
        "evaluation_status": label.get("evaluation_status", "eligible"), "reason": reason,
        "required_facts_json": json.dumps(label.get("required_facts", []), ensure_ascii=False),
        "minimum_sufficient_evidence_json": json.dumps(label.get("minimum_sufficient_evidence", []), ensure_ascii=False),
        "missing_information": str(label.get("missing_information", "")),
        "review_status": label.get("review_status", ""), "review_notes": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验证答案充分性标签和评测资格")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--review-size", type=int, default=15)
    args = parser.parse_args()

    labels = read_jsonl(args.labels)
    topics = read_topics(args.topics)
    evidence_by_id = {row["evidence_id"]: row for row in read_jsonl(args.evidence)}
    by_qid = {row["qid"]: row for row in labels}
    errors: list[str] = []
    if len(by_qid) != len(labels):
        errors.append("存在重复 qid")
    missing = sorted(set(topics) - set(by_qid))
    extra = sorted(set(by_qid) - set(topics))
    if missing:
        errors.append(f"缺少问题: {missing}")
    if extra:
        errors.append(f"未知问题: {extra}")

    eligible_fact_counts: list[int] = []
    incomplete_qids: list[str] = []
    excluded_qids: list[str] = []
    reviewed_qids: list[str] = []
    courses = Counter()
    query_types = Counter()
    review_candidates: dict[str, dict[str, str]] = {}

    for qid, label in sorted(by_qid.items()):
        if label.get("query") != topics.get(qid):
            errors.append(f"{qid}: labels 与 topics 的问题文本不一致")
        status = label.get("evaluation_status", "eligible")
        if status not in {"eligible", "excluded"}:
            errors.append(f"{qid}: 非法 evaluation_status={status}")
            continue
        if label.get("review_status") == "reviewed":
            reviewed_qids.append(qid)
        facts = label.get("required_facts", [])
        combinations = label.get("minimum_sufficient_evidence", [])
        complete = bool(label.get("evidence_complete"))
        query_types[label.get("query_type", "Unknown")] += 1
        course = primary_course(label, evidence_by_id)
        courses[course] += 1
        allowed = set(label.get("candidate_evidence_ids", []))

        if status == "excluded":
            excluded_qids.append(qid)
            if facts or combinations or complete:
                errors.append(f"{qid}: 已排除题目必须 facts/组合为空且 evidence_complete=false")
            if not str(label.get("exclusion_reason", "")).strip():
                errors.append(f"{qid}: 已排除题目缺少 exclusion_reason")
            continue

        eligible_fact_counts.append(len(facts))
        if not complete:
            incomplete_qids.append(qid)
            if label.get("review_status") != "reviewed":
                review_candidates[qid] = make_review_row(label, course, "证据不完整，需补资料、改写问题或排除评测")
        if complete and (not facts or not combinations):
            errors.append(f"{qid}: 证据完整但事实或最小证据组合为空")

        support_by_fact: dict[str, set[str]] = {}
        for index, item in enumerate(facts, 1):
            expected = f"F{index:02d}"
            if item.get("fact_id") != expected:
                errors.append(f"{qid}: fact_id 不连续")
            support = set(item.get("supporting_evidence_ids", []))
            if not support or not support <= allowed:
                errors.append(f"{qid}/{expected}: 证据为空或越界")
            if not support <= set(evidence_by_id):
                errors.append(f"{qid}/{expected}: 引用了不存在的证据")
            support_by_fact[expected] = support
        for combination in combinations:
            chosen = set(combination)
            if not chosen or not chosen <= allowed or not chosen <= set(evidence_by_id):
                errors.append(f"{qid}: 最小证据组合为空、越界或引用不存在")
                continue
            uncovered = [fact_id for fact_id, support in support_by_fact.items() if not support & chosen]
            if uncovered:
                errors.append(f"{qid}: 最小证据组合未覆盖 {uncovered}")
        if label.get("review_status") != "reviewed" and (len(facts) <= 1 or len(facts) >= 10):
            review_candidates.setdefault(qid, make_review_row(label, course, "事实数量异常，检查是否拆分过细或遗漏"))

    represented = {(row["primary_course"], row["query_type"]) for row in review_candidates.values()}
    for qid, label in sorted(by_qid.items()):
        if len(review_candidates) >= args.review_size:
            break
        if label.get("evaluation_status", "eligible") != "eligible" or label.get("review_status") == "reviewed":
            continue
        course = primary_course(label, evidence_by_id)
        key = (course, label.get("query_type", ""))
        if key in represented:
            continue
        represented.add(key)
        review_candidates[qid] = make_review_row(label, course, "学科与问题类型分层抽查")

    facts_stats = {
        "min": min(eligible_fact_counts) if eligible_fact_counts else 0,
        "median": statistics.median(eligible_fact_counts) if eligible_fact_counts else 0,
        "max": max(eligible_fact_counts) if eligible_fact_counts else 0,
        "mean": round(statistics.mean(eligible_fact_counts), 3) if eligible_fact_counts else 0,
    }
    report = {
        "schema_version": "stage2-sufficiency-verification-v2",
        "status": "PASS" if not errors else "FAIL", "queries_total": len(labels),
        "eligible_queries": len(labels) - len(excluded_qids), "excluded_queries": len(excluded_qids),
        "excluded_qids": excluded_qids, "reviewed_queries": len(reviewed_qids), "reviewed_qids": reviewed_qids,
        "facts_total_eligible": sum(eligible_fact_counts), "facts_per_eligible_query": facts_stats,
        "query_types": dict(sorted(query_types.items())), "primary_courses": dict(sorted(courses.items())),
        "incomplete_eligible_qids": incomplete_qids, "review_queue_size": len(review_candidates), "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "qid", "query", "primary_course", "query_type", "fact_count", "evidence_complete",
        "evaluation_status", "reason", "required_facts_json", "minimum_sufficient_evidence_json",
        "missing_information", "review_status", "review_notes",
    ]
    review_queue_path = args.review_queue
    try:
        handle = review_queue_path.open("w", encoding="utf-8-sig", newline="")
    except PermissionError:
        review_queue_path = args.review_queue.with_name(args.review_queue.stem + "_next.csv")
        handle = review_queue_path.open("w", encoding="utf-8-sig", newline="")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_candidates[qid] for qid in sorted(review_candidates))
    report["review_queue_path"] = str(review_queue_path)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
