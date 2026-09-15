from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TOPICS = PROJECT_DIR / "topics.csv"
DEFAULT_QRELS = PROJECT_DIR / "data" / "stage2" / "evidence" / "evidence_qrels.csv"
DEFAULT_EVIDENCE = PROJECT_DIR / "data" / "stage2" / "evidence" / "evidence_units.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "stage2" / "sufficiency" / "required_facts.jsonl"
DEFAULT_SCHEMA = PROJECT_DIR / "data" / "stage2" / "sufficiency" / "schema.json"
DEFAULT_MODEL = "deepseek-chat"
QUERY_TYPES = {"definition", "enumeration", "comparison", "mechanism", "synthesis"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"无法识别 CSV 编码: {path}") from last_error


def load_topics(path: Path) -> dict[str, str]:
    return {row["qid"]: row["query"] for row in read_csv_rows(path)}


def load_positive_qrels(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv_rows(path):
        raw_rel = (row.get("rel") or "").strip()
        if raw_rel and int(raw_rel) > 0:
            row["rel"] = int(raw_rel)
            grouped[row["qid"]].append(row)
    return grouped


def select_evidence(
    qrels: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    limit: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    candidates = []
    for row in sorted(qrels, key=lambda item: (-int(item["rel"]), item["evidence_id"])):
        evidence = evidence_by_id.get(row["evidence_id"])
        if not evidence:
            continue
        candidates.append(
            {
                "evidence_id": evidence["evidence_id"],
                "rel": int(row["rel"]),
                "course": evidence.get("course", ""),
                "chapter_path": evidence.get("chapter_path", ""),
                "document_title": evidence.get("document_title", ""),
                "text": evidence.get("text", "")[:max_chars],
            }
        )

    selected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for candidate in candidates:
        source = candidate["document_title"]
        if source not in seen_sources:
            selected.append(candidate)
            seen_sources.add(source)
        if len(selected) >= limit:
            return selected
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def extract_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型响应中没有 JSON 对象")
    return json.loads(text[start : end + 1])


def request_label(
    client: OpenAI,
    model: str,
    qid: str,
    query: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    system_prompt = (
        "你是计算机课程 RAG 评测标注员。只能依据给定证据，不得补充外部知识。"
        "目标是定义完整回答该问题所必须覆盖的原子事实，并标出每个事实的证据依据。"
        "query_type 只能是 definition、enumeration、comparison、mechanism、synthesis 之一。"
        "required_facts 必须是非空数组，每项包含 fact_id、description、supporting_evidence_ids。"
        "description 使用简洁中文；supporting_evidence_ids 必须来自输入。"
        "minimum_sufficient_evidence 是一个或多个证据 ID 数组，每个数组应能够共同覆盖全部 required_facts。"
        "若当前证据不能完整支持问题，设置 evidence_complete=false、required_facts=[]、minimum_sufficient_evidence=[]，"
        "并在 missing_information 中说明缺失内容。"
        "只返回严格 JSON 对象，不要 Markdown。"
    )
    payload = {"qid": qid, "query": query, "evidence": evidence}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
    )
    return extract_json_object(response.choices[0].message.content or "")


def validate_label(
    raw: dict[str, Any],
    qid: str,
    query: str,
    allowed_evidence_ids: set[str],
    model: str,
) -> dict[str, Any]:
    query_type = str(raw.get("query_type", "")).strip()
    if query_type not in QUERY_TYPES:
        raise ValueError(f"非法 query_type: {query_type}")

    evidence_complete = bool(raw.get("evidence_complete", True))
    facts = raw.get("required_facts")
    if not isinstance(facts, list):
        raise ValueError("required_facts 必须是数组")
    if not facts and evidence_complete:
        raise ValueError("证据完整时 required_facts 不能为空")

    normalized_facts = []
    covered_by_facts: set[str] = set()
    for index, fact in enumerate(facts, 1):
        if not isinstance(fact, dict):
            raise ValueError("required_facts 元素必须是对象")
        description = str(fact.get("description", "")).strip()
        support = list(dict.fromkeys(str(item) for item in fact.get("supporting_evidence_ids", [])))
        if not description or not support:
            if not evidence_complete:
                continue
            raise ValueError(f"第 {index} 个事实缺少描述或证据")
        unknown = set(support) - allowed_evidence_ids
        if unknown:
            raise ValueError(f"第 {index} 个事实引用未知证据: {sorted(unknown)}")
        normalized_facts.append(
            {
                "fact_id": f"F{index:02d}",
                "description": description,
                "supporting_evidence_ids": support,
            }
        )
        covered_by_facts.update(support)
    if evidence_complete and not normalized_facts:
        raise ValueError("证据完整时至少需要一个合法事实")

    combinations = raw.get("minimum_sufficient_evidence", [])
    normalized_combinations = []
    if not isinstance(combinations, list):
        raise ValueError("minimum_sufficient_evidence 必须是数组")
    for combination in combinations:
        if not isinstance(combination, list) or not combination:
            continue
        normalized = list(dict.fromkeys(str(item) for item in combination))
        unknown = set(normalized) - allowed_evidence_ids
        if unknown:
            raise ValueError(f"最小证据组合引用未知证据: {sorted(unknown)}")
        if not normalized_facts:
            continue
        supported_facts = {
            fact["fact_id"]
            for fact in normalized_facts
            if set(fact["supporting_evidence_ids"]) & set(normalized)
        }
        if len(supported_facts) == len(normalized_facts):
            normalized_combinations.append(normalized)

    missing_information = raw.get("missing_information", "")
    if isinstance(missing_information, list):
        missing_information = "；".join(str(item).strip() for item in missing_information if str(item).strip())
    elif isinstance(missing_information, dict):
        missing_information = json.dumps(missing_information, ensure_ascii=False, sort_keys=True)
    else:
        missing_information = str(missing_information).strip()

    return {
        "schema_version": "stage2-sufficiency-v1",
        "qid": qid,
        "query": query,
        "query_type": query_type,
        "required_facts": normalized_facts,
        "minimum_sufficient_evidence": normalized_combinations,
        "evidence_complete": evidence_complete,
        "missing_information": missing_information,
        "review_status": "auto",
        "evaluation_status": "eligible",
        "exclusion_reason": "",
        "annotation_model": model,
        "candidate_evidence_ids": sorted(allowed_evidence_ids),
    }


def save_jsonl_atomic(path: Path, rows_by_qid: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in sorted(rows_by_qid):
            handle.write(json.dumps(rows_by_qid[qid], ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp, path)


def write_schema(path: Path) -> None:
    schema = {
        "schema_version": "stage2-sufficiency-v1",
        "purpose": "Define answer-required atomic facts and minimum sufficient evidence for adaptive Parent-Child context evaluation.",
        "query_types": sorted(QUERY_TYPES),
        "review_status": ["auto", "reviewed"],
        "evaluation_status": ["eligible", "excluded"],
        "required_fields": [
            "qid",
            "query",
            "query_type",
            "required_facts",
            "minimum_sufficient_evidence",
            "evidence_complete",
            "review_status",
            "evaluation_status",
            "exclusion_reason",
        ],
        "fact_fields": ["fact_id", "description", "supporting_evidence_ids"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="用 DeepSeek 生成可断点续跑的答案要点与最小充分证据初标")
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-evidence", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    positive_qrels = load_positive_qrels(args.qrels)
    evidence_by_id = {row["evidence_id"]: row for row in read_jsonl(args.evidence)}
    missing_qids = sorted(set(topics) - set(positive_qrels))
    if missing_qids:
        raise ValueError(f"以下问题没有正例证据: {missing_qids}")

    existing = {}
    if args.output.exists() and not args.force:
        existing = {row["qid"]: row for row in read_jsonl(args.output)}
        for row in existing.values():
            if row.get("missing_information") == "[]":
                row["missing_information"] = ""
    pending = [qid for qid in sorted(topics) if args.force or qid not in existing]
    if args.limit_queries > 0:
        pending = pending[: args.limit_queries]

    print(f"问题总数: {len(topics)} | 已有初标: {len(existing)} | 本次待处理: {len(pending)}")
    if args.dry_run:
        for qid in pending:
            selected = select_evidence(
                positive_qrels[qid], evidence_by_id, args.max_evidence, args.max_chars
            )
            print(f"{qid}: 正例 {len(positive_qrels[qid])} | 送标 {len(selected)}")
        return

    write_schema(args.schema)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY 环境变量")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    failed = 0
    for index, qid in enumerate(pending, 1):
        selected = select_evidence(
            positive_qrels[qid], evidence_by_id, args.max_evidence, args.max_chars
        )
        print(f"[{index}/{len(pending)}] {qid}: 正例 {len(positive_qrels[qid])} | 送标 {len(selected)}")
        try:
            raw = request_label(client, args.model, qid, topics[qid], selected)
            label = validate_label(
                raw,
                qid,
                topics[qid],
                {item["evidence_id"] for item in selected},
                args.model,
            )
            existing[qid] = label
            save_jsonl_atomic(args.output, existing)
            print(
                f"  完成: {len(label['required_facts'])} 个事实 | "
                f"最小证据组合 {len(label['minimum_sufficient_evidence'])} 个"
            )
        except Exception as exc:
            failed += 1
            print(f"  失败，进度已保留: {exc}")
        time.sleep(max(0.0, args.delay))

    print(f"本次完成: {len(pending) - failed} | 失败: {failed} | 累计: {len(existing)}/{len(topics)}")


if __name__ == "__main__":
    main()
