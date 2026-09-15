#!/usr/bin/env python3
"""End-to-end answer comparison for Stage-1 chunks vs Parent-Child RRF parents."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from openai import OpenAI


SYSTEM_PROMPT = """你是计算机课程知识库问答助手。请严格遵守：
1. 只能依据给定资料回答，不得补充资料之外的事实；资料不足时明确说明。
2. 直接回答问题，结构清晰、简洁，保留必要的专业术语。
3. 每个关键结论后标注支持它的资料编号，如 [1] 或 [1][3]。
4. 不要引用没有实际支持对应结论的资料。使用中文回答。"""

JUDGE_PROMPT = """你是严格的 RAG 答案评测器。根据问题、必备事实、两组检索上下文，独立评价两个答案。
只输出一个 JSON 对象，不要 Markdown。格式必须为：
{
  "baseline": {
    "covered_fact_ids": ["F01"],
    "unsupported_claims": ["具体的无依据说法"],
    "citation_numbers": [1],
    "invalid_citation_numbers": [],
    "clarity_score": 1,
    "redundancy_score": 1
  },
  "parent_rrf": {同上},
  "preferred": "baseline|parent_rrf|tie",
  "reason": "一句话理由"
}
判定规则：
- covered_fact_ids：答案明确表达且语义正确的必备事实；不能因上下文包含而计入。
- unsupported_claims：答案中的实质性主张无法由该答案对应的上下文支持；无则为空数组。
- citation_numbers：答案实际出现的所有引用编号，去重。
- invalid_citation_numbers：编号越界，或引用片段不支持其紧邻主张的编号。
- clarity_score：1-5，5 最清晰。
- redundancy_score：1-5，5 最精炼、无重复。
- preferred：优先事实完整、依据充分、引用有效的答案；文风只作次要依据。"""

PRIORITY_QIDS = [
    "A004", "A008", "A010", "A011", "A033", "A039", "A054", "A059", "A063",
    "A001", "A003", "A015", "A020", "A024", "A047", "A058", "A061", "A064",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_run(path: Path) -> dict[str, list[str]]:
    runs: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6:
                continue
            qid = p[0]
            doc_id = " ".join(p[2:-3])
            runs[qid].append((int(p[-3]), doc_id))
    return {qid: [doc for _, doc in sorted(items)] for qid, items in runs.items()}


def load_parents(parent_dir: Path) -> dict[str, dict]:
    result = {}
    for path in sorted(parent_dir.glob("*_chunks.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            result[row["id"]] = row
    return result


def normalize_subject(value: str) -> str:
    v = (value or "").lower()
    if "operat" in v or v == "os":
        return "Operating System"
    if "network" in v or v == "cn":
        return "Computer Network"
    if "data structure" in v or v == "ds":
        return "Data Structure"
    return value or "Unknown"


def choose_qids(labels: dict[str, dict], baseline: dict[str, list[str]], parents: dict[str, dict], per_subject: int) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    order = PRIORITY_QIDS + sorted(q for q in labels if q not in PRIORITY_QIDS)
    for qid in order:
        if qid not in labels or not baseline.get(qid):
            continue
        top = parents.get(baseline[qid][0], {})
        subject = normalize_subject(top.get("course") or top.get("subject") or "")
        if subject in {"Operating System", "Computer Network", "Data Structure"}:
            groups[subject].append(qid)
    selected = []
    for subject in ("Operating System", "Computer Network", "Data Structure"):
        selected.extend(groups[subject][:per_subject])
    if len(selected) != per_subject * 3:
        raise RuntimeError(f"无法按学科选足题目: { {k: len(v) for k, v in groups.items()} }")
    return selected


def build_context(doc_ids: list[str], parents: dict[str, dict], top_k: int, max_chars: int) -> tuple[str, list[dict]]:
    blocks, meta = [], []
    for idx, doc_id in enumerate(doc_ids[:top_k], 1):
        row = parents.get(doc_id)
        if not row:
            raise KeyError(f"Parent 文本不存在: {doc_id}")
        course = row.get("course") or row.get("subject") or "Unknown"
        title = row.get("document_title") or row.get("source") or "Unknown"
        chapter = row.get("chapter_path") or "未标注"
        blocks.append(f"[{idx}] 课程：{course}；文档：{title}；章节：{chapter}\n{row.get('text', '')}")
        meta.append({"citation": idx, "doc_id": doc_id, "course": course, "chapter": chapter})
    context = "\n\n---\n\n".join(blocks)
    return context[:max_chars], meta


def call_text(client: OpenAI, model: str, question: str, context: str, retries: int) -> str:
    user = f"问题：\n{question}\n\n检索资料：\n{context}\n\n请给出最终答案。"
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_judge(client: OpenAI, model: str, payload: dict, retries: int) -> dict:
    user = json.dumps(payload, ensure_ascii=False)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return parse_json_object(response.choices[0].message.content)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def summarize(rows: list[dict]) -> dict:
    result = {"question_count": len(rows), "strategies": {}, "preference": defaultdict(int)}
    for strategy in ("baseline", "parent_rrf"):
        coverages, full, unsupported, cite_valid, clarity, redundancy = [], [], [], [], [], []
        for row in rows:
            total = len(row["required_facts"])
            judged = row["judgement"][strategy]
            covered = set(judged.get("covered_fact_ids", []))
            cov = len(covered) / total if total else 1.0
            coverages.append(cov)
            full.append(cov == 1.0)
            unsupported.append(len(judged.get("unsupported_claims", [])))
            citations = set(judged.get("citation_numbers", []))
            invalid = set(judged.get("invalid_citation_numbers", []))
            cite_valid.append((len(citations - invalid) / len(citations)) if citations else 0.0)
            clarity.append(float(judged.get("clarity_score", 0)))
            redundancy.append(float(judged.get("redundancy_score", 0)))
        result["strategies"][strategy] = {
            "mean_fact_coverage": safe_mean(coverages),
            "full_fact_rate": safe_mean([float(x) for x in full]),
            "mean_unsupported_claims": safe_mean(unsupported),
            "mean_citation_validity": safe_mean(cite_valid),
            "mean_clarity_score": safe_mean(clarity),
            "mean_redundancy_score": safe_mean(redundancy),
        }
    for row in rows:
        result["preference"][row["judgement"].get("preferred", "unknown")] += 1
    result["preference"] = dict(result["preference"])
    return result


def write_markdown(path: Path, summary: dict, rows: list[dict]) -> None:
    b = summary["strategies"]["baseline"]
    p = summary["strategies"]["parent_rrf"]
    lines = [
        "# Stage 2 端到端答案质量对比", "",
        f"- 题目数：{summary['question_count']}（三学科各 5 题）",
        "- 条件：相同 DeepSeek 模型、提示词、Top-5、上下文字符预算",
        "- 说明：LLM 自动评审用于低成本筛查，重要结论仍建议人工抽检。", "",
        "| 指标 | 原 Chunk 策略 | Parent RRF | 差值 |", "|---|---:|---:|---:|",
    ]
    metrics = [
        ("必备事实覆盖率", "mean_fact_coverage"), ("完整事实答案率", "full_fact_rate"),
        ("平均无依据主张数", "mean_unsupported_claims"), ("引用有效率", "mean_citation_validity"),
        ("清晰度（1-5）", "mean_clarity_score"), ("精炼度（1-5）", "mean_redundancy_score"),
    ]
    for label, key in metrics:
        lines.append(f"| {label} | {b[key]:.4f} | {p[key]:.4f} | {p[key]-b[key]:+.4f} |")
    lines += ["", f"偏好计数：`{json.dumps(summary['preference'], ensure_ascii=False)}`", "", "## 逐题结果", ""]
    for row in rows:
        j = row["judgement"]
        lines.append(f"- {row['qid']}（{row['subject']}）：{j.get('preferred')} — {j.get('reason', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-context-chars", type=int, default=16000)
    ap.add_argument("--per-subject", type=int, default=5)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    root = args.project_root.resolve()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY")

    labels_path = root / "data/stage2/sufficiency/required_facts.jsonl"
    baseline_path = root / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run"
    rrf_path = root / "experiments/stage2/parent_child/minilm_c200_o50_parent_rrf20_eval60/parent_child_parents.run"
    parent_dir = root / "data/stage2/parent_child/minilm_c200_o50_v1/parents"
    out_dir = root / "experiments/stage2/answer_eval/parent_rrf20_vs_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "results.jsonl"

    labels = {x["qid"]: x for x in load_jsonl(labels_path) if x.get("evaluation_status") == "eligible"}
    baseline = load_run(baseline_path)
    rrf = load_run(rrf_path)
    parents = load_parents(parent_dir)
    selected = choose_qids(labels, baseline, parents, args.per_subject)
    existing = {x["qid"]: x for x in load_jsonl(rows_path)} if rows_path.exists() else {}
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for pos, qid in enumerate(selected, 1):
        if qid in existing:
            print(f"[{pos:02d}/{len(selected)}] {qid}: 使用缓存", flush=True)
            continue
        label = labels[qid]
        base_ctx, base_meta = build_context(baseline[qid], parents, args.top_k, args.max_context_chars)
        rrf_ctx, rrf_meta = build_context(rrf[qid], parents, args.top_k, args.max_context_chars)
        subject = normalize_subject(parents[baseline[qid][0]].get("course") or parents[baseline[qid][0]].get("subject"))
        print(f"[{pos:02d}/{len(selected)}] {qid} ({subject}): 生成 baseline", flush=True)
        base_answer = call_text(client, args.model, label["query"], base_ctx, args.retries)
        print(f"[{pos:02d}/{len(selected)}] {qid}: 生成 parent_rrf", flush=True)
        rrf_answer = call_text(client, args.model, label["query"], rrf_ctx, args.retries)
        judge_payload = {
            "question": label["query"], "required_facts": label["required_facts"],
            "baseline": {"context": base_ctx, "answer": base_answer},
            "parent_rrf": {"context": rrf_ctx, "answer": rrf_answer},
        }
        print(f"[{pos:02d}/{len(selected)}] {qid}: 自动评审", flush=True)
        judgement = call_judge(client, args.model, judge_payload, args.retries)
        row = {
            "qid": qid, "subject": subject, "query": label["query"],
            "required_facts": label["required_facts"],
            "baseline": {"retrieval": base_meta, "answer": base_answer},
            "parent_rrf": {"retrieval": rrf_meta, "answer": rrf_answer},
            "judgement": judgement,
        }
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        existing[qid] = row

    ordered = [existing[qid] for qid in selected]
    summary = summarize(ordered)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", summary, ordered)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"结果目录: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
