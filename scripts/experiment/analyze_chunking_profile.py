"""Profile corpus/evidence lengths and derive a data-driven chunk search plan.

This command is read-only with respect to production data.  It writes an
experiment profile and a frozen baseline snapshot under the requested output
directory.  Candidate values come from observed quantiles rather than a fixed
100/200/300/400 grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_prep.build_parent_child_chunks import (  # noqa: E402
    nonempty_line_spans,
    sentence_spans,
    token_offsets,
)
from scripts.pipeline.corpus_files import discover_chunk_files  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def nearest_rank(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return int(ordered[index])


def distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0, "p95": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "p25": nearest_rank(values, 0.25),
        "p50": nearest_rank(values, 0.50),
        "p75": nearest_rank(values, 0.75),
        "p90": nearest_rank(values, 0.90),
        "p95": nearest_rank(values, 0.95),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
    }


def round_to_step(value: int, step: int = 5) -> int:
    return max(step, int(step * round(value / step)))


def derive_candidates(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive candidates from structural quantiles and effective model capacity."""
    sentences = profile["sentence_tokens"]
    paragraphs = profile["paragraph_tokens"]
    capacity = profile["embedding_capacity"]
    raw_targets = [paragraphs["p90"], paragraphs["p95"], capacity["shared_safe_body_tokens"]]
    safe_capacity = int(capacity["shared_safe_body_tokens"])
    targets = sorted(
        {min(round_to_step(int(value), 10), safe_capacity) for value in raw_targets if value}
    )
    overlap_limits = sorted(
        {0, *(
            round_to_step(int(value), 5)
            for value in (sentences["p50"], sentences["p75"])
            if value
        )}
    )
    candidates: list[dict[str, Any]] = []
    for target in targets:
        minimum = min(target, round_to_step(int(paragraphs["p75"]), 5))
        maximum = target
        for overlap in overlap_limits:
            candidates.append(
                {
                    "strategy": "structured",
                    "target_tokens": target,
                    "min_tokens": minimum,
                    "max_tokens": maximum,
                    "overlap_max_tokens": min(overlap, max(0, target - 1)),
                    "overlap_ratio_cap": round(min(0.49, overlap / target), 3) if overlap else 0.0,
                    "derived_from": {
                        "target": "paragraph p90/p95 and shared safe embedding capacity",
                        "minimum": "paragraph p75",
                        "maximum": "embedding max sequence minus metadata and special tokens",
                        "overlap": "no overlap or sentence p50/p75",
                    },
                }
            )
    return candidates


def derive_refinement_candidates(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive a coarse-to-fine raw-text search around structural spans and model capacity.

    Legacy course Children are embedded without metadata prefixes, so their upper
    bound is the raw body capacity rather than the smaller cross-corpus bound used
    for prefixed AI records.
    """
    sentences = profile["sentence_tokens"]
    paragraphs = profile["paragraph_tokens"]
    raw_capacity = int(profile["embedding_capacity"]["raw_body_tokens"])
    structural_targets = (
        int(paragraphs["p95"]) + int(sentences["p75"]),
        int(paragraphs["p95"]) + int(sentences["p95"]),
        raw_capacity,
    )
    targets = sorted({min(round_to_step(value, 5), raw_capacity) for value in structural_targets})
    overlap = min(round_to_step(int(sentences["p75"]), 5), raw_capacity - 1)
    minimum = min(round_to_step(int(paragraphs["p75"]), 5), min(targets))
    candidates: list[dict[str, Any]] = []
    for target in targets:
        for overlap_max in (0, overlap):
            candidates.append(
                {
                    "strategy": "structured",
                    "target_tokens": target,
                    "min_tokens": minimum,
                    "max_tokens": target,
                    "overlap_max_tokens": min(overlap_max, target - 1),
                    "overlap_ratio_cap": round(min(0.25, overlap_max / target), 3) if overlap_max else 0.0,
                    "derived_from": {
                        "target": "paragraph p95 + sentence p75/p95, plus raw embedding capacity",
                        "minimum": "paragraph p75",
                        "maximum": "raw embedding capacity (no metadata prefix)",
                        "overlap": "none or sentence p75, capped at 25%",
                    },
                }
            )
    return candidates


def derive_hybrid_candidates(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive longer candidates for a hybrid Dense/BM25/BGE retrieval chain.

    Dense truncation is a soft system boundary because BM25 and the cross-encoder
    still consume the complete Child.  The upper probe is therefore bounded at
    twice the dense body capacity, while intermediate probes add observed
    structural spans to that capacity.
    """
    sentences = profile["sentence_tokens"]
    paragraphs = profile["paragraph_tokens"]
    raw_capacity = int(profile["embedding_capacity"]["raw_body_tokens"])
    targets = sorted(
        {
            round_to_step(raw_capacity + int(paragraphs["p75"]), 5),
            round_to_step(raw_capacity + int(paragraphs["p95"]) + int(sentences["p75"]), 5),
            round_to_step(2 * raw_capacity, 5),
        }
    )
    overlaps = sorted(
        {
            round_to_step(int(sentences["p75"]), 5),
            round_to_step(int(paragraphs["p95"]), 5),
        }
    )
    minimum = round_to_step(int(paragraphs["p75"]), 5)
    return [
        {
            "strategy": "structured",
            "target_tokens": target,
            "min_tokens": min(minimum, target),
            "max_tokens": target,
            "overlap_max_tokens": min(overlap, target - 1),
            "overlap_ratio_cap": round(min(0.25, overlap / target), 3),
            "derived_from": {
                "target": "dense body capacity + structural quantiles, bounded by 2x dense capacity",
                "minimum": "paragraph p75",
                "maximum": "soft hybrid bound; BM25 and BGE read beyond dense truncation",
                "overlap": "sentence p75 or paragraph p95, capped at 25%",
            },
        }
        for target in targets
        for overlap in overlaps
    ]


def iter_records(directory: Path) -> Iterable[dict[str, Any]]:
    for filename in discover_chunk_files(directory):
        yield from load_json(directory / filename)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents-dir", type=Path, default=ROOT / "data/stage4/parent_child/minilm_c200_o50_v1/parents")
    parser.add_argument("--children-dir", type=Path, default=ROOT / "data/stage4/parent_child/minilm_c200_o50_v1/children")
    parser.add_argument("--evidence-units", type=Path, default=ROOT / "data/stage2/evidence/evidence_units.jsonl")
    parser.add_argument("--index-config", type=Path, default=ROOT / "vector_db/stage1/minilm/raw/kb_config.json")
    parser.add_argument("--production-config", type=Path, default=ROOT / "config/parent_child.json")
    parser.add_argument("--holdout-summary", type=Path, default=ROOT / "experiments/stage2/answer_eval/adaptive_holdout15_v1/summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/chunking_optimization")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    model_name = str(load_json(args.index_config.resolve())["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.offline)
    tokenizer.model_max_length = 1_000_000_000
    embedding_model = SentenceTransformer(model_name, local_files_only=args.offline)
    embedding_max_seq_length = int(embedding_model.max_seq_length)
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    parent_lengths: list[int] = []
    paragraph_lengths: list[int] = []
    sentence_lengths: list[int] = []
    subjects: Counter[str] = Counter()
    parents = list(iter_records(args.parents_dir.resolve()))
    for parent in parents:
        text = str(parent.get("text") or "")
        parent_lengths.append(len(token_offsets(tokenizer, text)))
        subjects[str(parent.get("subject") or "unknown")] += 1
        for start, end in nonempty_line_spans(text):
            paragraph_lengths.append(len(token_offsets(tokenizer, text[start:end])))
            sentence_lengths.extend(
                len(token_offsets(tokenizer, text[sentence_start:sentence_end]))
                for sentence_start, sentence_end in sentence_spans(text, start, end)
            )
    evidence_lengths = [
        len(token_offsets(tokenizer, str(unit.get("text") or "")))
        for unit in load_jsonl(args.evidence_units.resolve())
    ]
    context_prefix_lengths: list[int] = []
    for child in iter_records(args.children_dir.resolve()):
        retrieval_context = str(child.get("retrieval_context") or "").strip()
        if retrieval_context:
            context_prefix_lengths.append(
                len(token_offsets(tokenizer, retrieval_context + "\n正文："))
            )
    profile = {
        "schema_version": "chunking-profile-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tokenizer": model_name,
        "parents_dir": str(args.parents_dir.resolve()),
        "parent_count": len(parents),
        "subjects": dict(subjects),
        "parent_tokens": distribution(parent_lengths),
        "paragraph_tokens": distribution(paragraph_lengths),
        "sentence_tokens": distribution(sentence_lengths),
        "evidence_tokens": distribution(evidence_lengths),
        "context_prefix_tokens": distribution(context_prefix_lengths),
        "embedding_capacity": {
            "model_max_seq_length": embedding_max_seq_length,
            "special_tokens": special_tokens,
            "raw_body_tokens": embedding_max_seq_length - special_tokens,
            "shared_safe_body_tokens": embedding_max_seq_length
            - special_tokens
            - (nearest_rank(context_prefix_lengths, 0.95) if context_prefix_lengths else 0),
            "worst_case_body_tokens": embedding_max_seq_length
            - special_tokens
            - (max(context_prefix_lengths) if context_prefix_lengths else 0),
            "note": "Shared-safe capacity uses the observed p95 metadata prefix; worst-case capacity is reported separately instead of letting one outlier collapse the search space.",
        },
    }
    profile["candidates"] = derive_candidates(profile)
    profile["refinement_candidates"] = derive_refinement_candidates(profile)
    profile["hybrid_candidates"] = derive_hybrid_candidates(profile)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    (output / "corpus_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    production = load_json(args.production_config.resolve())
    holdout = load_json(args.holdout_summary.resolve())
    baseline = {
        "schema_version": "chunking-baseline-v1",
        "production_config_path": str(args.production_config.resolve()),
        "production_config_sha256": digest(args.production_config.resolve()),
        "production_config": production,
        "holdout_summary_path": str(args.holdout_summary.resolve()),
        "holdout_summary_sha256": digest(args.holdout_summary.resolve()),
        "holdout": holdout,
        "selection_policy": "Development data selects one candidate; frozen holdout is run once and never used to retune.",
    }
    (output / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"profile": profile, "baseline_saved": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
