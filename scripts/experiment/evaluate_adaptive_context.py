#!/usr/bin/env python3
"""Uniformly re-judge an adaptive combination of completed context experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path[:0] = [str(root), str(root / "scripts/experiment")]
    from evaluate_answer_quality import build_context as build_plain, load_parents, load_run, parse_json_object  # noqa: E402
    from evaluate_context_aware import aggregate, build_context as build_highlighted, judge, load_children  # noqa: E402
    from openai import OpenAI  # noqa: E402

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY")
    baseline_run = load_run(root / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run")
    parents = load_parents(root / "data/stage2/parent_child/minilm_c200_o50_v1/parents")
    children, _ = load_children(root / "data/stage2/parent_child/minilm_c200_o50_v1/children")
    labels = {row["qid"]: row for row in load_jsonl(root / "data/stage2/sufficiency/required_facts.jsonl")}
    base_rows = load_jsonl(root / "experiments/stage2/answer_eval/parent_rrf20_vs_baseline/results.jsonl")
    rescue_rows = {row["qid"]: row for row in load_jsonl(root / "experiments/stage2/answer_eval/parent_rrf20_rescue1_vs_baseline/results.jsonl")}
    highlighted_rows = {row["qid"]: row for row in load_jsonl(root / "experiments/stage2/answer_eval/parent_rrf20_rescue1_child_highlight_vs_baseline/results.jsonl")}
    out_dir = root / "experiments/stage2/answer_eval/adaptive_definition_highlight_vs_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"
    done = {row["qid"]: row for row in load_jsonl(out_path)} if out_path.exists() else {}
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    for index, base in enumerate(base_rows, 1):
        qid = base["qid"]
        if qid in done:
            print(f"[{index:02d}/15] {qid}: 使用缓存", flush=True)
            continue
        query_type = labels[qid].get("query_type", "")
        baseline_context, _ = build_plain(baseline_run[qid], parents, 5, 16000)
        if query_type == "definition":
            source = highlighted_rows[qid]
            parent_ids = [item["doc_id"] for item in source["context_aware"]["retrieval"]]
            highlights = {}
            for item in source["context_aware"]["retrieval"]:
                child_id = item.get("highlight_child_id")
                if child_id and child_id in children:
                    highlights[item["doc_id"]] = {**children[child_id], "rerank_score": item.get("highlight_score")}
            adaptive_context, retrieval = build_highlighted(parent_ids, parents, highlights, 16000)
            adaptive_answer = source["context_aware"]["answer"]
            variant = "child_highlight_parent"
        else:
            source = rescue_rows[qid]
            parent_ids = [item["doc_id"] for item in source["rescue"]["retrieval"]]
            adaptive_context, retrieval = build_plain(parent_ids, parents, len(parent_ids), 16000)
            adaptive_answer = source["rescue"]["answer"]
            variant = "parent_rescue"
        payload = {
            "question": base["query"], "required_facts": base["required_facts"],
            "baseline": {"context": baseline_context, "answer": base["baseline"]["answer"]},
            "context_aware": {"context": adaptive_context, "answer": adaptive_answer},
        }
        print(f"[{index:02d}/15] {qid}: 统一复评（{variant}）", flush=True)
        judgement = judge(client, args.model, payload, parse_json_object, args.retries)
        row = {
            "qid": qid, "subject": base["subject"], "query": base["query"], "query_type": query_type,
            "required_facts": base["required_facts"], "baseline": base["baseline"],
            "adaptive": {"variant": variant, "retrieval": retrieval, "answer": adaptive_answer},
            "judgement": judgement,
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[qid] = row

    # aggregate() expects the candidate data under judgement.context_aware.
    rows = [done[row["qid"]] for row in base_rows]
    summary = {"question_count": len(rows), "strategies": {"baseline": aggregate(rows, "baseline"), "adaptive": aggregate(rows, "context_aware")}}
    preferences: dict[str, int] = defaultdict(int)
    for row in rows:
        preferences[row["judgement"].get("preferred", "unknown")] += 1
    summary["preference"] = dict(preferences)
    thresholds = {
        "fact_coverage": summary["strategies"]["adaptive"]["mean_fact_coverage"] >= 0.8778,
        "full_fact_rate": summary["strategies"]["adaptive"]["full_fact_rate"] >= 11 / 15,
        "citation_validity": summary["strategies"]["adaptive"]["mean_citation_validity"] >= 0.95,
        "unsupported_claims": summary["strategies"]["adaptive"]["mean_unsupported_claims"] <= summary["strategies"]["baseline"]["mean_unsupported_claims"],
    }
    by_qid = {row["qid"]: row for row in rows}
    required_counts = {"A033": 5, "A054": 8, "A063": 5}
    thresholds["critical_questions"] = all(len(set(by_qid[qid]["judgement"]["context_aware"].get("covered_fact_ids", []))) >= count for qid, count in required_counts.items())
    summary["development_acceptance"] = thresholds
    summary["development_pass"] = all(thresholds.values())
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    b, a = summary["strategies"]["baseline"], summary["strategies"]["adaptive"]
    metrics = [("必备事实覆盖率", "mean_fact_coverage"), ("完整事实答案率", "full_fact_rate"), ("平均无依据主张数", "mean_unsupported_claims"), ("引用有效率", "mean_citation_validity"), ("清晰度（1-5）", "mean_clarity_score"), ("精炼度（1-5）", "mean_redundancy_score")]
    md = ["# 自适应 Context Assembly 开发集评测", "", "- definition：Child 高亮 + Parent", "- enumeration / comparison：Parent RRF Top-5 + BGE 补位", "", "| 指标 | 原 Chunk | Adaptive | 差值 |", "|---|---:|---:|---:|"]
    for label, metric in metrics:
        md.append(f"| {label} | {b[metric]:.4f} | {a[metric]:.4f} | {a[metric]-b[metric]:+.4f} |")
    md += ["", f"开发集通过：**{summary['development_pass']}**", "", f"门槛：`{json.dumps(thresholds, ensure_ascii=False)}`", "", "## 逐题结果", ""]
    for row in rows:
        j = row["judgement"]
        total = len(row["required_facts"])
        bc = len(set(j["baseline"].get("covered_fact_ids", [])))
        ac = len(set(j["context_aware"].get("covered_fact_ids", [])))
        md.append(f"- {row['qid']} [{row['adaptive']['variant']}]：{bc}/{total} → {ac}/{total}；{j.get('preferred')} — {j.get('reason', '')}")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"结果目录: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
