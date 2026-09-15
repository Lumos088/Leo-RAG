#!/usr/bin/env python3
"""Head-to-head answer evaluation for a chunking candidate versus production.

The development questions and production answers are reused from the frozen
Stage-2 evaluation.  Only the candidate answer is generated again, which keeps
the comparison paired and reduces API cost.  Holdout qids are rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_evaluation_scope(reference_qids: set[str], holdout: set[str], scope: str) -> list[str]:
    overlap = sorted(reference_qids & holdout)
    if scope == "development" and overlap:
        raise RuntimeError(f"development set overlaps sealed holdout: {overlap}")
    if scope == "holdout" and reference_qids != holdout:
        raise RuntimeError(
            "holdout evaluation requires reference qids to exactly match the sealed manifest"
        )
    return overlap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-results", type=Path, default=ROOT / "experiments/stage2/answer_eval/adaptive_definition_highlight_vs_baseline/results.jsonl")
    parser.add_argument("--baseline-run", type=Path, default=ROOT / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run")
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--children-dir", type=Path, required=True)
    parser.add_argument("--reference-children-dir", type=Path, default=ROOT / "data/stage2/parent_child/minilm_c200_o50_v1/children")
    parser.add_argument("--holdout-manifest", type=Path, default=ROOT / "experiments/stage2/answer_eval/adaptive_holdout15_v1/holdout_manifest.json")
    parser.add_argument(
        "--scope", choices=("development", "holdout"), default="development",
        help="Development rejects holdout overlap; holdout requires an exact manifest match.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=16000)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    sys.path[:0] = [str(ROOT), str(ROOT / "scripts/experiment")]
    from app.reranker import BGECrossEncoderReranker
    from evaluate_answer_quality import build_context as build_plain, call_text, load_parents, load_run, parse_json_object
    from evaluate_context_aware import aggregate, build_context as build_highlighted, judge, load_children, select_highlights
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    references = read_jsonl(args.reference_results.resolve())
    holdout = set(json.loads(args.holdout_manifest.resolve().read_text(encoding="utf-8"))["qids"])
    reference_qids = {row["qid"] for row in references}
    leaked = validate_evaluation_scope(reference_qids, holdout, args.scope)

    baseline_run = load_run(args.baseline_run.resolve())
    candidate_run = load_run(args.candidate_run.resolve())
    parents = load_parents(args.parent_dir.resolve())
    _, children_by_parent = load_children(args.children_dir.resolve())
    reference_children, _ = load_children(args.reference_children_dir.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.jsonl"
    done = {row["qid"]: row for row in read_jsonl(output_path)} if output_path.exists() else {}
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    reranker = BGECrossEncoderReranker()

    for position, reference in enumerate(references, 1):
        qid = reference["qid"]
        if qid in done:
            print(f"[{position:02d}/{len(references)}] {qid}: cached", flush=True)
            continue
        ranked = list(candidate_run[qid][: args.top_k])
        rescue = next((doc_id for doc_id in baseline_run[qid] if doc_id not in ranked), None)
        if rescue:
            baseline_rank = baseline_run[qid].index(rescue)
            ranked.insert(min(baseline_rank, len(ranked)), rescue)
        query_type = reference.get("query_type", "enumeration")
        if query_type == "definition":
            highlights = select_highlights(reference["query"], ranked, children_by_parent, reranker)
            context, retrieval = build_highlighted(ranked, parents, highlights, args.max_context_chars)
            variant = "child_highlight_parent"
        else:
            context, retrieval = build_plain(ranked, parents, len(ranked), args.max_context_chars)
            variant = "parent_rescue"
        print(f"[{position:02d}/{len(references)}] {qid}: generate candidate", flush=True)
        started = time.perf_counter()
        answer = call_text(client, args.model, reference["query"], context, args.retries)
        generation_seconds = time.perf_counter() - started
        reference_payload = reference["adaptive"]
        reference_ids = [item["doc_id"] for item in reference_payload["retrieval"]]
        if reference_payload.get("variant") == "child_highlight_parent":
            reference_highlights = {}
            for item in reference_payload["retrieval"]:
                child_id = item.get("highlight_child_id")
                if child_id and child_id in reference_children:
                    reference_highlights[item["doc_id"]] = reference_children[child_id]
            reference_context, _ = build_highlighted(
                reference_ids, parents, reference_highlights, args.max_context_chars
            )
        else:
            reference_context, _ = build_plain(
                reference_ids, parents, len(reference_ids), args.max_context_chars
            )
        judgement = judge(
            client,
            args.model,
            {
                "question": reference["query"],
                "required_facts": reference["required_facts"],
                "baseline": {"context": reference_context, "answer": reference_payload["answer"]},
                "context_aware": {"context": context, "answer": answer},
            },
            parse_json_object,
            args.retries,
        )
        row = {
            "qid": qid,
            "subject": reference["subject"],
            "query": reference["query"],
            "query_type": query_type,
            "required_facts": reference["required_facts"],
            "reference": reference_payload,
            "candidate": {
                "variant": variant,
                "retrieval": retrieval,
                "rescue_parent_id": rescue,
                "context": context,
                "context_chars": len(context),
                "generation_seconds": generation_seconds,
                "answer": answer,
            },
            "judgement": judgement,
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[qid] = row

    rows = [done[row["qid"]] for row in references]
    normalized = [
        {**row, "baseline": row["reference"], "context_aware": row["candidate"]}
        for row in rows
    ]
    reference_metrics = aggregate(normalized, "baseline")
    candidate_metrics = aggregate(normalized, "context_aware")
    preference: dict[str, int] = defaultdict(int)
    for row in rows:
        preference[row["judgement"].get("preferred", "unknown")] += 1
    acceptance = {
        "fact_coverage_not_lower": candidate_metrics["mean_fact_coverage"] >= reference_metrics["mean_fact_coverage"],
        "full_fact_rate_not_lower": candidate_metrics["full_fact_rate"] >= reference_metrics["full_fact_rate"],
        "citation_validity_at_least_095": candidate_metrics["mean_citation_validity"] >= 0.95,
        "unsupported_not_higher": candidate_metrics["mean_unsupported_claims"] <= reference_metrics["mean_unsupported_claims"],
        "mean_context_within_budget": mean(row["candidate"]["context_chars"] for row in rows) <= args.max_context_chars,
    }
    summary = {
        "schema_version": "chunking-answer-head-to-head-v1",
        "question_count": len(rows),
        "scope": args.scope,
        "qids": [row["qid"] for row in rows],
        "holdout_overlap": leaked,
        "reference": reference_metrics,
        "candidate": candidate_metrics,
        "preference": dict(preference),
        "mean_candidate_context_chars": mean(row["candidate"]["context_chars"] for row in rows),
        "mean_candidate_generation_seconds": mean(row["candidate"]["generation_seconds"] for row in rows),
        "acceptance": acceptance,
        "pass": all(acceptance.values()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
