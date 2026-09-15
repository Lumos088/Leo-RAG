"""Prepare and evaluate data-derived structure-aware Child chunk variants.

Production configuration is never modified.  Every artifact is isolated under
``data/experiments``, ``vector_db/experiments`` and ``experiments``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "experiments/chunking_optimization/corpus_profile.json"
HOLDOUT_MANIFEST = ROOT / "experiments/stage2/answer_eval/adaptive_holdout15_v1/holdout_manifest.json"


def slug(candidate: dict[str, Any]) -> str:
    return (
        f"struct_t{candidate['target_tokens']}_min{candidate['min_tokens']}"
        f"_max{candidate['max_tokens']}_o{candidate['overlap_max_tokens']}"
    )


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def write_runtime_config(
    candidate: dict[str, Any], data_dir: Path, index_dir: Path, path: Path
) -> None:
    config = json.loads((ROOT / "config/parent_child_stage2_reference.json").read_text(encoding="utf-8"))
    config.update(
        {
            "enabled": True,
            "variant": slug(candidate),
            "child_index_dir": relative(index_dir),
            "parent_dir": relative(data_dir / "parents"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(
    profile: dict[str, Any], variants: list[dict[str, Any]], output: Path, filename: str
) -> None:
    rows = []
    for candidate in variants:
        name = slug(candidate)
        result_dir = output / "results" / name
        comparison_path = result_dir / "comparison.json"
        diagnosis_path = result_dir / "failure_analysis.json"
        if not comparison_path.is_file() or not diagnosis_path.is_file():
            continue
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
        pc = comparison["parent_child"]
        fact5 = diagnosis["aggregate"]["fact_metrics"]["@5"]
        index_config = json.loads(
            (ROOT / "vector_db/experiments/chunking_optimization" / name / "kb_config.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "variant": name,
                "parameters": candidate,
                "strict_fact_recall_at_5": fact5["parent_child_strict_mean_fact_recall"],
                "all_facts_covered_at_5": fact5["parent_child_all_facts_covered_rate"],
                "evidence_recall_at_5": pc["evidence"]["EvidenceRecall@5"],
                "parent_ndcg_at_10": pc["parent"]["ParentNDCG@10"],
                "success_at_1_proxy": pc["parent"]["ParentMRR@10"],
                "child_duplicate_parent_rate_at_50": pc["retrieval_shape"]["ChildDuplicateParentRate@50"],
                "p95_query_seconds": pc["timing"]["p95_seconds"],
                "children": index_config["ntotal"],
                "index_bytes": index_config["index_bytes"],
            }
        )
    rows.sort(
        key=lambda row: (
            -row["all_facts_covered_at_5"],
            -row["strict_fact_recall_at_5"],
            row["child_duplicate_parent_rate_at_50"],
            row["p95_query_seconds"],
        )
    )
    report = {
        "schema_version": "chunking-search-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_scope": "development/evaluation pool only; frozen answer holdout not accessed",
        "candidate_source": str(PROFILE),
        "model_limit": profile["embedding_capacity"],
        "ranked_variants": rows,
        "shortlist": [row["variant"] for row in rows[:3]],
    }
    (output / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--candidate-set",
        choices=("candidates", "refinement_candidates", "hybrid_candidates"),
        default="candidates",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not any((args.prepare, args.index, args.evaluate)):
        parser.error("select at least one of --prepare, --index, --evaluate")

    profile = json.loads(args.profile.resolve().read_text(encoding="utf-8"))
    variants = profile[args.candidate_set][: args.limit]
    output = ROOT / "experiments/chunking_optimization"
    for number, candidate in enumerate(variants, 1):
        name = slug(candidate)
        print(f"[{number}/{len(variants)}] {name}", flush=True)
        data_dir = ROOT / "data/experiments/chunking_optimization" / name
        index_dir = ROOT / "vector_db/experiments/chunking_optimization" / name
        config_path = output / "configs" / f"{name}.json"
        result_dir = output / "results" / name
        if args.prepare and (args.force or not (data_dir / "report.json").is_file()):
            run(
                [sys.executable, "scripts/data_prep/build_parent_child_chunks.py", "--strategy", "structured",
                 "--target-tokens", str(candidate["target_tokens"]), "--min-tokens", str(candidate["min_tokens"]),
                 "--max-tokens", str(candidate["max_tokens"]), "--overlap-tokens", str(candidate["overlap_max_tokens"]),
                 "--overlap-ratio-cap", str(candidate["overlap_ratio_cap"]), "--output-dir", str(data_dir), "--offline"]
            )
        if args.index and (args.force or not (index_dir / "kb.index").is_file()):
            run(
                [sys.executable, "scripts/data_prep/build_vector_db.py", "--model", "minilm", "--chunks-dir",
                 str(data_dir / "children"), "--output-dir", str(index_dir), "--text-mode", "raw",
                 "--batch-size", "128", "--device", args.device, "--overwrite"]
            )
        write_runtime_config(candidate, data_dir, index_dir, config_path)
        if args.evaluate and (args.force or not (result_dir / "failure_analysis.json").is_file()):
            run(
                [sys.executable, "scripts/experiment/evaluate_parent_child.py", "--config", str(config_path),
                 "--baseline-run", str(ROOT / "experiments/stage2/baseline_eval60/minilm-raw-rerank-eval60-v2_rerank.run"),
                 "--exclude-qids", str(HOLDOUT_MANIFEST), "--device", args.device,
                 "--output-dir", str(result_dir)]
            )
            run(
                [sys.executable, "scripts/experiment/diagnose_parent_child.py", "--pc-dir", str(result_dir),
                 "--pc-data", str(data_dir), "--exclude-qids", str(HOLDOUT_MANIFEST)]
            )
    if args.evaluate:
        filename = (
            "retrieval_screening.json"
            if args.candidate_set == "candidates"
            else f"retrieval_screening_{args.candidate_set.removesuffix('_candidates')}.json"
        )
        summarize(profile, variants, output, filename)


if __name__ == "__main__":
    main()
