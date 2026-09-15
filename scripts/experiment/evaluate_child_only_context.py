#!/usr/bin/env python3
"""Evaluate Parent routing with top-2 Child-only generation context."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean


def overlap_ratio(a: dict, b: dict) -> float:
    a0, a1 = int(a.get("source_char_start", 0)), int(a.get("source_char_end", 0))
    b0, b1 = int(b.get("source_char_start", 0)), int(b.get("source_char_end", 0))
    intersection = max(0, min(a1, b1) - max(a0, b0))
    return intersection / max(1, min(a1 - a0, b1 - b0))


def select_children(query: str, parent_ids: list[str], by_parent: dict[str, list[dict]], reranker, per_parent: int) -> dict[str, list[dict]]:
    candidates = []
    for parent_id in parent_ids:
        for child in by_parent.get(parent_id, []):
            candidates.append({**child, "score": 0.0})
    ranked = reranker.rerank(query, candidates, top_k=len(candidates)) if candidates else []
    selected: dict[str, list[dict]] = defaultdict(list)
    for child in ranked:
        parent_id = child["parent_id"]
        current = selected[parent_id]
        if len(current) >= per_parent:
            continue
        if current and any(overlap_ratio(child, chosen) > 0.50 for chosen in current):
            continue
        current.append(child)
    # Very short Parents may not have enough non-overlapping children.
    for parent_id in parent_ids:
        if len(selected[parent_id]) >= per_parent:
            continue
        ranked_parent = [item for item in ranked if item["parent_id"] == parent_id]
        for child in ranked_parent:
            if child not in selected[parent_id]:
                selected[parent_id].append(child)
            if len(selected[parent_id]) >= per_parent:
                break
    return dict(selected)


def build_child_context(parent_ids: list[str], parents: dict[str, dict], selected: dict[str, list[dict]], max_chars: int) -> tuple[str, list[dict]]:
    blocks, metadata = [], []
    for number, parent_id in enumerate(parent_ids, 1):
        parent = parents[parent_id]
        course = parent.get("course") or parent.get("subject") or "Unknown"
        title = parent.get("document_title") or parent.get("source") or "Unknown"
        chapter = parent.get("chapter_path") or "未标注"
        children = selected.get(parent_id, [])
        snippets = "\n\n".join(f"命中片段 {idx}：\n{child.get('text', '')}" for idx, child in enumerate(children, 1))
        blocks.append(f"[{number}] 课程：{course}；文档：{title}；章节：{chapter}\n{snippets}")
        metadata.append({
            "citation": number, "parent_id": parent_id, "course": course, "chapter": chapter,
            "child_ids": [child["id"] for child in children],
            "child_scores": [child.get("rerank_score") for child in children],
        })
    context = "\n\n---\n\n".join(blocks)
    return context[:max_chars], metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--children-per-parent", type=int, default=2)
    ap.add_argument("--max-context-chars", type=int, default=16000)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    root = args.project_root.resolve()
    sys.path[:0] = [str(root), str(root / "scripts/experiment")]
    from app.reranker import BGECrossEncoderReranker  # noqa: E402
    from evaluate_answer_quality import build_context as build_baseline_context, call_text, load_parents, load_run, parse_json_object  # noqa: E402
    from evaluate_context_aware import aggregate, insert_rescue, judge, jsonl, load_children  # noqa: E402
    from openai import OpenAI  # noqa: E402

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY")
    prior = jsonl(root / "experiments/stage2/answer_eval/parent_rrf20_vs_baseline/results.jsonl")
    if len(prior) != 15:
        raise RuntimeError(f"需要 15 道既有基线，实际 {len(prior)}")
    base_run = load_run(root / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run")
    rrf_run = load_run(root / "experiments/stage2/parent_child/minilm_c200_o50_parent_rrf20_eval60/parent_child_parents.run")
    parents = load_parents(root / "data/stage2/parent_child/minilm_c200_o50_v1/parents")
    _, by_parent = load_children(root / "data/stage2/parent_child/minilm_c200_o50_v1/children")
    out_dir = root / f"experiments/stage2/answer_eval/parent_rrf20_rescue1_child{args.children_per_parent}_only_vs_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.jsonl"
    done = {row["qid"]: row for row in jsonl(result_path)}
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    print("加载本地 BGE Child 选择器...", flush=True)
    reranker = BGECrossEncoderReranker()

    for index, source in enumerate(prior, 1):
        qid = source["qid"]
        if qid in done:
            print(f"[{index:02d}/15] {qid}: 使用缓存", flush=True)
            continue
        parent_ids, rescue_id = insert_rescue(rrf_run[qid], base_run[qid], args.top_k)
        chosen = select_children(source["query"], parent_ids, by_parent, reranker, args.children_per_parent)
        context, retrieval = build_child_context(parent_ids, parents, chosen, args.max_context_chars)
        baseline_context, _ = build_baseline_context(base_run[qid], parents, args.top_k, args.max_context_chars)
        print(f"[{index:02d}/15] {qid}: 生成 Child-only Top-{args.children_per_parent}", flush=True)
        started = time.perf_counter()
        answer = call_text(client, args.model, source["query"], context, args.retries)
        generation_seconds = time.perf_counter() - started
        payload = {
            "question": source["query"], "required_facts": source["required_facts"],
            "baseline": {"context": baseline_context, "answer": source["baseline"]["answer"]},
            "context_aware": {"context": context, "answer": answer},
        }
        print(f"[{index:02d}/15] {qid}: 自动评审", flush=True)
        judgement = judge(client, args.model, payload, parse_json_object, args.retries)
        row = {
            "qid": qid, "subject": source["subject"], "query": source["query"], "required_facts": source["required_facts"],
            "baseline": source["baseline"],
            "context_aware": {"retrieval": retrieval, "rescue_parent_id": rescue_id, "context_chars": len(context), "generation_seconds": generation_seconds, "answer": answer},
            "judgement": judgement,
        }
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[qid] = row

    rows = [done[row["qid"]] for row in prior]
    summary = {
        "question_count": len(rows), "children_per_parent": args.children_per_parent,
        "strategies": {"baseline": aggregate(rows, "baseline"), "context_aware": aggregate(rows, "context_aware")},
        "mean_context_chars": mean(row["context_aware"]["context_chars"] for row in rows),
        "mean_generation_seconds": mean(row["context_aware"]["generation_seconds"] for row in rows),
    }
    preferences: dict[str, int] = defaultdict(int)
    for row in rows:
        preferences[row["judgement"].get("preferred", "unknown")] += 1
    summary["preference"] = dict(preferences)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    b, c = summary["strategies"]["baseline"], summary["strategies"]["context_aware"]
    metrics = [("必备事实覆盖率", "mean_fact_coverage"), ("完整事实答案率", "full_fact_rate"), ("平均无依据主张数", "mean_unsupported_claims"), ("引用有效率", "mean_citation_validity"), ("清晰度（1-5）", "mean_clarity_score"), ("精炼度（1-5）", "mean_redundancy_score")]
    md = ["# Parent 路由 + Child-only Context 实验", "", f"- 每个 Parent 最多保留 {args.children_per_parent} 个低重叠 Child", "- Parent 只负责召回、排序、去重和引用归属，不重复发送 Parent 全文", f"- 平均上下文字符数：{summary['mean_context_chars']:.1f}", f"- 平均生成耗时：{summary['mean_generation_seconds']:.3f} 秒", "", "| 指标 | 原 Chunk | Child-only | 差值 |", "|---|---:|---:|---:|"]
    for label, metric in metrics:
        md.append(f"| {label} | {b[metric]:.4f} | {c[metric]:.4f} | {c[metric]-b[metric]:+.4f} |")
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
