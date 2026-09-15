#!/usr/bin/env python3
"""Evaluate Parent-RRF Top-5 plus one baseline evidence-rescue parent."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean


JUDGE_PROMPT = """你是严格的 RAG 答案评测器。根据问题、必备事实及各自检索上下文，独立评价 baseline 和 rescue 两个答案。
只输出 JSON 对象，不要 Markdown：
{
  "baseline": {"covered_fact_ids": ["F01"], "unsupported_claims": [], "citation_numbers": [1], "invalid_citation_numbers": [], "clarity_score": 1, "redundancy_score": 1},
  "rescue": {"covered_fact_ids": ["F01"], "unsupported_claims": [], "citation_numbers": [1], "invalid_citation_numbers": [], "clarity_score": 1, "redundancy_score": 1},
  "preferred": "baseline|rescue|tie",
  "reason": "一句话理由"
}
covered_fact_ids 只统计答案明确、正确表达的必备事实。unsupported_claims 只统计对应上下文无法支持的实质性主张。
invalid_citation_numbers 包括编号越界或不能支持紧邻主张的引用。clarity_score 为 1-5，5 最清晰；redundancy_score 为 1-5，5 最精炼。
优先事实完整、依据充分且引用有效的答案，文风仅作次要依据。"""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def rescue_doc_ids(rrf_ids: list[str], baseline_ids: list[str], top_k: int = 5, slots: int = 1) -> list[str]:
    selected = list(rrf_ids[:top_k])
    added = 0
    for doc_id in baseline_ids:
        if doc_id not in selected:
            selected.append(doc_id)
            added += 1
            if added >= slots:
                break
    return selected


def call_judge(client, model: str, payload: dict, parse_json_object, retries: int) -> dict:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return parse_json_object(response.choices[0].message.content)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def score(rows: list[dict], strategy: str) -> dict:
    coverage, full, unsupported, citation_validity, clarity, concise = [], [], [], [], [], []
    for row in rows:
        judged = row["judgement"][strategy]
        total = len(row["required_facts"])
        cov = len(set(judged.get("covered_fact_ids", []))) / total if total else 1.0
        coverage.append(cov)
        full.append(float(cov == 1.0))
        unsupported.append(len(judged.get("unsupported_claims", [])))
        cited = set(judged.get("citation_numbers", []))
        invalid = set(judged.get("invalid_citation_numbers", []))
        citation_validity.append(len(cited - invalid) / len(cited) if cited else 0.0)
        clarity.append(float(judged.get("clarity_score", 0)))
        concise.append(float(judged.get("redundancy_score", 0)))
    avg = lambda xs: mean(xs) if xs else 0.0
    return {
        "mean_fact_coverage": avg(coverage),
        "full_fact_rate": avg(full),
        "mean_unsupported_claims": avg(unsupported),
        "mean_citation_validity": avg(citation_validity),
        "mean_clarity_score": avg(clarity),
        "mean_redundancy_score": avg(concise),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rescue-slots", type=int, default=1)
    ap.add_argument("--max-context-chars", type=int, default=16000)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "scripts/experiment"))
    from evaluate_answer_quality import (  # noqa: E402
        build_context, call_text, load_parents, load_run, parse_json_object,
    )
    from openai import OpenAI  # noqa: E402

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY")
    source_dir = root / "experiments/stage2/answer_eval/parent_rrf20_vs_baseline"
    source_rows = load_jsonl(source_dir / "results.jsonl")
    if len(source_rows) != 15:
        raise RuntimeError(f"应复用 15 道基线结果，实际为 {len(source_rows)}")
    baseline_run = load_run(root / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run")
    rrf_run = load_run(root / "experiments/stage2/parent_child/minilm_c200_o50_parent_rrf20_eval60/parent_child_parents.run")
    parents = load_parents(root / "data/stage2/parent_child/minilm_c200_o50_v1/parents")
    out_dir = root / "experiments/stage2/answer_eval/parent_rrf20_rescue1_vs_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    completed = {x["qid"]: x for x in load_jsonl(out_path)}
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    for pos, source in enumerate(source_rows, 1):
        qid = source["qid"]
        if qid in completed:
            print(f"[{pos:02d}/15] {qid}: 使用缓存", flush=True)
            continue
        ids = rescue_doc_ids(rrf_run[qid], baseline_run[qid], args.top_k, args.rescue_slots)
        context, retrieval = build_context(ids, parents, len(ids), args.max_context_chars)
        print(f"[{pos:02d}/15] {qid}: 生成 rescue（{len(ids)} Parents）", flush=True)
        answer = call_text(client, args.model, source["query"], context, args.retries)
        # Rebuild the exact baseline context so citation evaluation is grounded.
        base_context, _ = build_context(baseline_run[qid], parents, args.top_k, args.max_context_chars)
        payload = {
            "question": source["query"], "required_facts": source["required_facts"],
            "baseline": {"context": base_context, "answer": source["baseline"]["answer"]},
            "rescue": {"context": context, "answer": answer},
        }
        print(f"[{pos:02d}/15] {qid}: 自动评审", flush=True)
        judgement = call_judge(client, args.model, payload, parse_json_object, args.retries)
        row = {
            "qid": qid, "subject": source["subject"], "query": source["query"],
            "required_facts": source["required_facts"], "baseline": source["baseline"],
            "rescue": {"retrieval": retrieval, "answer": answer}, "judgement": judgement,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        completed[qid] = row

    rows = [completed[x["qid"]] for x in source_rows]
    summary = {
        "question_count": len(rows), "top_k": args.top_k, "rescue_slots": args.rescue_slots,
        "strategies": {"baseline": score(rows, "baseline"), "rescue": score(rows, "rescue")},
        "preference": dict(defaultdict(int)),
    }
    pref: dict[str, int] = defaultdict(int)
    for row in rows:
        pref[row["judgement"].get("preferred", "unknown")] += 1
    summary["preference"] = dict(pref)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    b, r = summary["strategies"]["baseline"], summary["strategies"]["rescue"]
    labels = [
        ("必备事实覆盖率", "mean_fact_coverage"), ("完整事实答案率", "full_fact_rate"),
        ("平均无依据主张数", "mean_unsupported_claims"), ("引用有效率", "mean_citation_validity"),
        ("清晰度（1-5）", "mean_clarity_score"), ("精炼度（1-5）", "mean_redundancy_score"),
    ]
    md = ["# Parent RRF 证据补位实验", "", "- Parent RRF Top-5 + 原 BGE 首个未重复 Parent（共 6 个上下文块）", "- 原策略答案复用，生成参数与上一轮一致", "", "| 指标 | 原 Chunk | Parent RRF + Rescue | 差值 |", "|---|---:|---:|---:|"]
    for label, key_name in labels:
        md.append(f"| {label} | {b[key_name]:.4f} | {r[key_name]:.4f} | {r[key_name]-b[key_name]:+.4f} |")
    md += ["", f"偏好计数：`{json.dumps(summary['preference'], ensure_ascii=False)}`", "", "## 逐题结果", ""]
    for row in rows:
        j = row["judgement"]
        total = len(row["required_facts"])
        bc = len(set(j["baseline"].get("covered_fact_ids", [])))
        rc = len(set(j["rescue"].get("covered_fact_ids", [])))
        md.append(f"- {row['qid']}：事实 {bc}/{total} → {rc}/{total}；{j.get('preferred')} — {j.get('reason', '')}")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"结果目录: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
