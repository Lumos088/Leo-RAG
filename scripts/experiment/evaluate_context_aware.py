#!/usr/bin/env python3
"""Evaluate evidence-aware Parent/Child context assembly against the baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean


JUDGE_PROMPT = """你是严格的 RAG 答案评测器。根据问题、必备事实及各自检索上下文，独立评价 baseline 和 context_aware 两个答案。
只输出 JSON，不要 Markdown：
{
  "baseline": {"covered_fact_ids": ["F01"], "unsupported_claims": [], "citation_numbers": [1], "invalid_citation_numbers": [], "clarity_score": 1, "redundancy_score": 1},
  "context_aware": {"covered_fact_ids": ["F01"], "unsupported_claims": [], "citation_numbers": [1], "invalid_citation_numbers": [], "clarity_score": 1, "redundancy_score": 1},
  "preferred": "baseline|context_aware|tie", "reason": "一句话理由"
}
covered_fact_ids 只统计答案明确且正确表达的必备事实。unsupported_claims 只统计对应上下文无法支持的实质性主张。
invalid_citation_numbers 包括编号越界或不能支持紧邻主张的引用。clarity_score 1-5，5 最清晰；redundancy_score 1-5，5 最精炼。
优先事实完整、依据充分且引用有效的答案，文风仅作次要依据。"""


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_children(path: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    by_id: dict[str, dict] = {}
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for file in sorted(path.glob("*_chunks.json")):
        for child in json.loads(file.read_text(encoding="utf-8")):
            by_id[child["id"]] = child
            by_parent[child["parent_id"]].append(child)
    return by_id, by_parent


def insert_rescue(rrf_ids: list[str], baseline_ids: list[str], top_k: int) -> tuple[list[str], str | None]:
    selected = list(rrf_ids[:top_k])
    rescue = next((doc for doc in baseline_ids if doc not in selected), None)
    if rescue:
        baseline_rank = baseline_ids.index(rescue)
        selected.insert(min(baseline_rank, len(selected)), rescue)
    return selected, rescue


def select_highlights(query: str, parent_ids: list[str], by_parent: dict[str, list[dict]], reranker) -> dict[str, dict]:
    candidates = []
    for parent_id in parent_ids:
        for child in by_parent.get(parent_id, []):
            candidates.append({"id": child["id"], "parent_id": parent_id, "text": child.get("text", ""), "score": 0.0})
    if not candidates:
        return {}
    ranked = reranker.rerank(query, candidates, top_k=len(candidates))
    best: dict[str, dict] = {}
    for item in ranked:
        best.setdefault(item["parent_id"], item)
    return best


def build_context(parent_ids: list[str], parents: dict[str, dict], highlights: dict[str, dict], max_chars: int) -> tuple[str, list[dict]]:
    blocks, metadata = [], []
    for number, parent_id in enumerate(parent_ids, 1):
        parent = parents[parent_id]
        course = parent.get("course") or parent.get("subject") or "Unknown"
        title = parent.get("document_title") or parent.get("source") or "Unknown"
        chapter = parent.get("chapter_path") or "未标注"
        child = highlights.get(parent_id)
        highlight = child.get("text", "") if child else ""
        block = (
            f"[{number}] 课程：{course}；文档：{title}；章节：{chapter}\n"
            f"关键命中片段：\n{highlight}\n\n"
            f"所属 Parent 完整上下文：\n{parent.get('text', '')}"
        )
        blocks.append(block)
        metadata.append({
            "citation": number, "doc_id": parent_id,
            "highlight_child_id": child.get("id") if child else None,
            "highlight_score": child.get("rerank_score") if child else None,
            "course": course, "chapter": chapter,
        })
    raw = "\n\n---\n\n".join(blocks)
    return raw[:max_chars], metadata


def judge(client, model: str, payload: dict, parser, retries: int) -> dict:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return parser(response.choices[0].message.content)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def aggregate(rows: list[dict], name: str) -> dict:
    vectors: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        item = row["judgement"][name]
        total = len(row["required_facts"])
        coverage = len(set(item.get("covered_fact_ids", []))) / total if total else 1.0
        citations = set(item.get("citation_numbers", []))
        invalid = set(item.get("invalid_citation_numbers", []))
        vectors["mean_fact_coverage"].append(coverage)
        vectors["full_fact_rate"].append(float(coverage == 1.0))
        vectors["mean_unsupported_claims"].append(float(len(item.get("unsupported_claims", []))))
        vectors["mean_citation_validity"].append(len(citations - invalid) / len(citations) if citations else 0.0)
        vectors["mean_clarity_score"].append(float(item.get("clarity_score", 0)))
        vectors["mean_redundancy_score"].append(float(item.get("redundancy_score", 0)))
    return {key: mean(values) for key, values in vectors.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-context-chars", type=int, default=16000)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts/experiment"))
    from app.reranker import BGECrossEncoderReranker  # noqa: E402
    from evaluate_answer_quality import call_text, load_parents, load_run, parse_json_object  # noqa: E402
    from openai import OpenAI  # noqa: E402

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY")
    source_rows = jsonl(root / "experiments/stage2/answer_eval/parent_rrf20_vs_baseline/results.jsonl")
    if len(source_rows) != 15:
        raise RuntimeError(f"需要上一轮 15 题结果，实际 {len(source_rows)}")
    baseline_run = load_run(root / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run")
    rrf_run = load_run(root / "experiments/stage2/parent_child/minilm_c200_o50_parent_rrf20_eval60/parent_child_parents.run")
    parents = load_parents(root / "data/stage2/parent_child/minilm_c200_o50_v1/parents")
    _, by_parent = load_children(root / "data/stage2/parent_child/minilm_c200_o50_v1/children")
    out_dir = root / "experiments/stage2/answer_eval/parent_rrf20_rescue1_child_highlight_vs_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done = {row["qid"]: row for row in jsonl(out_path)}
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    print("加载本地 BGE Child 证据选择器...", flush=True)
    reranker = BGECrossEncoderReranker()

    for position, source in enumerate(source_rows, 1):
        qid = source["qid"]
        if qid in done:
            print(f"[{position:02d}/15] {qid}: 使用缓存", flush=True)
            continue
        parent_ids, rescue_id = insert_rescue(rrf_run[qid], baseline_run[qid], args.top_k)
        highlights = select_highlights(source["query"], parent_ids, by_parent, reranker)
        context, retrieval = build_context(parent_ids, parents, highlights, args.max_context_chars)
        from evaluate_answer_quality import build_context as build_plain_context  # noqa: E402
        base_context, _ = build_plain_context(baseline_run[qid], parents, args.top_k, args.max_context_chars)
        print(f"[{position:02d}/15] {qid}: 生成 context-aware（{len(parent_ids)} Parents + Child highlights）", flush=True)
        started = time.perf_counter()
        answer = call_text(client, args.model, source["query"], context, args.retries)
        latency = time.perf_counter() - started
        payload = {
            "question": source["query"], "required_facts": source["required_facts"],
            "baseline": {"context": base_context, "answer": source["baseline"]["answer"]},
            "context_aware": {"context": context, "answer": answer},
        }
        print(f"[{position:02d}/15] {qid}: 自动评审", flush=True)
        judgement = judge(client, args.model, payload, parse_json_object, args.retries)
        row = {
            "qid": qid, "subject": source["subject"], "query": source["query"], "required_facts": source["required_facts"],
            "baseline": source["baseline"], "context_aware": {"retrieval": retrieval, "rescue_parent_id": rescue_id, "context_chars": len(context), "generation_seconds": latency, "answer": answer},
            "judgement": judgement,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[qid] = row

    rows = [done[row["qid"]] for row in source_rows]
    summary = {"question_count": len(rows), "strategies": {"baseline": aggregate(rows, "baseline"), "context_aware": aggregate(rows, "context_aware")}}
    pref: dict[str, int] = defaultdict(int)
    for row in rows:
        pref[row["judgement"].get("preferred", "unknown")] += 1
    summary["preference"] = dict(pref)
    summary["mean_context_chars"] = mean(row["context_aware"]["context_chars"] for row in rows)
    summary["mean_generation_seconds"] = mean(row["context_aware"]["generation_seconds"] for row in rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    b, c = summary["strategies"]["baseline"], summary["strategies"]["context_aware"]
    metrics = [("必备事实覆盖率", "mean_fact_coverage"), ("完整事实答案率", "full_fact_rate"), ("平均无依据主张数", "mean_unsupported_claims"), ("引用有效率", "mean_citation_validity"), ("清晰度（1-5）", "mean_clarity_score"), ("精炼度（1-5）", "mean_redundancy_score")]
    md = ["# 证据感知 Context Assembly 实验", "", "- Parent RRF Top-5 + 1 个按原 BGE 名次插入的补位 Parent", "- 每个 Parent 前置一个由本地 BGE 选出的最佳 Child 片段", f"- 平均上下文字符数：{summary['mean_context_chars']:.1f}", f"- 平均生成耗时：{summary['mean_generation_seconds']:.3f} 秒", "", "| 指标 | 原 Chunk | Context-aware | 差值 |", "|---|---:|---:|---:|"]
    for label, key_name in metrics:
        md.append(f"| {label} | {b[key_name]:.4f} | {c[key_name]:.4f} | {c[key_name]-b[key_name]:+.4f} |")
    md += ["", f"偏好计数：`{json.dumps(summary['preference'], ensure_ascii=False)}`", "", "## 逐题结果", ""]
    for row in rows:
        j = row["judgement"]
        total = len(row["required_facts"])
        bc = len(set(j["baseline"].get("covered_fact_ids", [])))
        cc = len(set(j["context_aware"].get("covered_fact_ids", [])))
        md.append(f"- {row['qid']}：事实 {bc}/{total} → {cc}/{total}；{j.get('preferred')} — {j.get('reason', '')}")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"结果目录: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
