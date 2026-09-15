#!/usr/bin/env python3
"""Balanced unseen holdout evaluation for the Stage-2 adaptive context policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_query_type(query: str) -> str:
    comparison_markers = ("区别", "比较", "对比", "差异", "异同", "优缺点")
    enumeration_markers = ("哪些", "几种", "两种", "三种", "四种", "五种", "六种", "七种", "八种", "九种", "十种", "枚举", "列出", "包括")
    if any(marker in query for marker in comparison_markers) or ("和" in query and "特性" in query):
        return "comparison"
    if any(marker in query for marker in enumeration_markers) or any(f"{n}个" in query for n in "一二两三四五六七八九十123456789"):
        return "enumeration"
    if any(marker in query for marker in ("什么", "定义", "解释")):
        return "definition"
    return "enumeration"


def normalize_subject(value: str) -> str:
    value = (value or "").lower()
    if "operat" in value or value == "os":
        return "Operating System"
    if "network" in value or value == "cn":
        return "Computer Network"
    if "data structure" in value or value == "ds":
        return "Data Structure"
    return "Unknown"


def choose_holdout(labels: dict[str, dict], dev_qids: set[str], baseline_run: dict[str, list[str]], parents: dict[str, dict], per_subject: int) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for qid, label in labels.items():
        if qid in dev_qids or label.get("evaluation_status") != "eligible" or not baseline_run.get(qid):
            continue
        parent = parents[baseline_run[qid][0]]
        grouped[normalize_subject(parent.get("course") or parent.get("subject"))].append(qid)
    output = []
    for subject in ("Operating System", "Computer Network", "Data Structure"):
        ranked = sorted(grouped[subject], key=lambda qid: hashlib.sha256(f"stage2-holdout-v1:{qid}".encode()).hexdigest())
        if len(ranked) < per_subject:
            raise RuntimeError(f"{subject} 只能选出 {len(ranked)} 题")
        output.extend(ranked[:per_subject])
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--per-subject", type=int, default=5)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    root = args.project_root.resolve()
    sys.path[:0] = [str(root), str(root / "scripts/experiment")]
    from app.reranker import BGECrossEncoderReranker  # noqa: E402
    from evaluate_answer_quality import build_context as build_plain, call_text, load_parents, load_run, parse_json_object  # noqa: E402
    from evaluate_context_aware import aggregate, build_context as build_highlighted, insert_rescue, judge, load_children, select_highlights  # noqa: E402
    from openai import OpenAI  # noqa: E402

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY")
    labels = {row["qid"]: row for row in read_jsonl(root / "data/stage2/sufficiency/required_facts.jsonl")}
    dev_rows = read_jsonl(root / "experiments/stage2/answer_eval/parent_rrf20_vs_baseline/results.jsonl")
    dev_qids = {row["qid"] for row in dev_rows}
    base_run = load_run(root / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run")
    rrf_run = load_run(root / "experiments/stage2/parent_child/minilm_c200_o50_parent_rrf20_eval60/parent_child_parents.run")
    parents = load_parents(root / "data/stage2/parent_child/minilm_c200_o50_v1/parents")
    _, by_parent = load_children(root / "data/stage2/parent_child/minilm_c200_o50_v1/children")
    selected = choose_holdout(labels, dev_qids, base_run, parents, args.per_subject)
    out_dir = root / "experiments/stage2/answer_eval/adaptive_holdout15_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "holdout_manifest.json").write_text(json.dumps({"selection_seed": "stage2-holdout-v1", "qids": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path = out_dir / "results.jsonl"
    done = {row["qid"]: row for row in read_jsonl(out_path)}
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    print(f"冻结 Holdout: {selected}", flush=True)
    print("加载本地 BGE Child 选择器...", flush=True)
    reranker = BGECrossEncoderReranker()

    for index, qid in enumerate(selected, 1):
        if qid in done:
            print(f"[{index:02d}/15] {qid}: 使用缓存", flush=True)
            continue
        label = labels[qid]
        query, query_type = label["query"], infer_query_type(label["query"])
        baseline_context, baseline_meta = build_plain(base_run[qid], parents, 5, 16000)
        parent_ids, rescue_id = insert_rescue(rrf_run[qid], base_run[qid], 5)
        if query_type == "definition":
            highlights = select_highlights(query, parent_ids, by_parent, reranker)
            adaptive_context, adaptive_meta = build_highlighted(parent_ids, parents, highlights, 16000)
            variant = "child_highlight_parent"
        else:
            adaptive_context, adaptive_meta = build_plain(parent_ids, parents, len(parent_ids), 16000)
            variant = "parent_rescue"
        print(f"[{index:02d}/15] {qid} ({query_type}): 生成 baseline", flush=True)
        started = time.perf_counter()
        baseline_answer = call_text(client, args.model, query, baseline_context, args.retries)
        baseline_seconds = time.perf_counter() - started
        print(f"[{index:02d}/15] {qid}: 生成 adaptive", flush=True)
        started = time.perf_counter()
        adaptive_answer = call_text(client, args.model, query, adaptive_context, args.retries)
        adaptive_seconds = time.perf_counter() - started
        payload = {
            "question": query, "required_facts": label["required_facts"],
            "baseline": {"context": baseline_context, "answer": baseline_answer},
            "context_aware": {"context": adaptive_context, "answer": adaptive_answer},
        }
        print(f"[{index:02d}/15] {qid}: 自动评审", flush=True)
        judgement = judge(client, args.model, payload, parse_json_object, args.retries)
        subject = normalize_subject(parents[base_run[qid][0]].get("course") or parents[base_run[qid][0]].get("subject"))
        row = {
            "qid": qid, "subject": subject, "query": query, "query_type": query_type, "label_query_type": label.get("query_type"), "required_facts": label["required_facts"],
            "baseline": {"retrieval": baseline_meta, "context_chars": len(baseline_context), "generation_seconds": baseline_seconds, "answer": baseline_answer},
            "adaptive": {"variant": variant, "retrieval": adaptive_meta, "rescue_parent_id": rescue_id, "context_chars": len(adaptive_context), "generation_seconds": adaptive_seconds, "answer": adaptive_answer},
            "judgement": judgement,
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[qid] = row

    rows = [done[qid] for qid in selected]
    summary = {
        "question_count": len(rows), "qids": selected,
        "strategies": {"baseline": aggregate(rows, "baseline"), "adaptive": aggregate(rows, "context_aware")},
        "mean_context_chars": {"baseline": mean(row["baseline"]["context_chars"] for row in rows), "adaptive": mean(row["adaptive"]["context_chars"] for row in rows)},
        "mean_generation_seconds": {"baseline": mean(row["baseline"]["generation_seconds"] for row in rows), "adaptive": mean(row["adaptive"]["generation_seconds"] for row in rows)},
    }
    pref: dict[str, int] = defaultdict(int)
    for row in rows:
        pref[row["judgement"].get("preferred", "unknown")] += 1
    summary["preference"] = dict(pref)
    b, a = summary["strategies"]["baseline"], summary["strategies"]["adaptive"]
    acceptance = {
        "fact_coverage_not_lower": a["mean_fact_coverage"] >= b["mean_fact_coverage"],
        "full_fact_rate_not_lower": a["full_fact_rate"] >= b["full_fact_rate"],
        "citation_validity": a["mean_citation_validity"] >= 0.95,
        "unsupported_not_higher": a["mean_unsupported_claims"] <= b["mean_unsupported_claims"],
        "context_budget": summary["mean_context_chars"]["adaptive"] <= 16000,
    }
    summary["holdout_acceptance"] = acceptance
    summary["holdout_pass"] = all(acceptance.values())
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = [("必备事实覆盖率", "mean_fact_coverage"), ("完整事实答案率", "full_fact_rate"), ("平均无依据主张数", "mean_unsupported_claims"), ("引用有效率", "mean_citation_validity"), ("清晰度（1-5）", "mean_clarity_score"), ("精炼度（1-5）", "mean_redundancy_score")]
    md = ["# 自适应 Context Assembly 冻结 Holdout", "", f"- 题目：{', '.join(selected)}", "- 三学科各 5 题；选题在生成答案前按固定哈希冻结", "", "| 指标 | 原 Chunk | Adaptive | 差值 |", "|---|---:|---:|---:|"]
    for label_name, metric in metrics:
        md.append(f"| {label_name} | {b[metric]:.4f} | {a[metric]:.4f} | {a[metric]-b[metric]:+.4f} |")
    md += ["", f"Holdout 通过：**{summary['holdout_pass']}**", "", f"门槛：`{json.dumps(acceptance, ensure_ascii=False)}`", "", "## 逐题结果", ""]
    for row in rows:
        j = row["judgement"]
        total = len(row["required_facts"])
        bc = len(set(j["baseline"].get("covered_fact_ids", [])))
        ac = len(set(j["context_aware"].get("covered_fact_ids", [])))
        md.append(f"- {row['qid']} [{row['query_type']}/{row['adaptive']['variant']}]：{bc}/{total} → {ac}/{total}；{j.get('preferred')} — {j.get('reason', '')}")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"结果目录: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
