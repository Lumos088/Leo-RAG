"""Create a first Parent-Child corpus while preserving every Stage-1 parent ID."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "parent-child-v1"

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.pipeline.corpus_files import discover_chunk_files


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def nonempty_line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        content_end = len(line.rstrip("\r\n"))
        content = line[:content_end]
        left = len(content) - len(content.lstrip())
        right = len(content.rstrip())
        if right > left:
            spans.append((cursor + left, cursor + right))
        cursor += len(line)
    if text and not text.endswith(("\n", "\r")) and not spans:
        stripped = text.strip()
        if stripped:
            start = text.index(stripped)
            spans.append((start, start + len(stripped)))
    return spans


def token_offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    return [(int(start), int(end)) for start, end in encoded["offset_mapping"] if end > start]


def atomic_units(tokenizer, text: str, hard_max_tokens: int) -> list[dict[str, int]]:
    units: list[dict[str, int]] = []
    for line_start, line_end in nonempty_line_spans(text):
        line = text[line_start:line_end]
        offsets = token_offsets(tokenizer, line)
        if not offsets:
            continue
        for token_start in range(0, len(offsets), hard_max_tokens):
            window = offsets[token_start : token_start + hard_max_tokens]
            units.append(
                {
                    "start": line_start + window[0][0],
                    "end": line_start + window[-1][1],
                    "tokens": len(window),
                }
            )
    return units


SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+(?=[A-Z0-9])")


def sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return trimmed sentence spans inside an existing source span."""
    segment = text[start:end]
    spans: list[tuple[int, int]] = []
    cursor = 0
    boundaries = [match.end() for match in SENTENCE_BOUNDARY.finditer(segment)] + [len(segment)]
    for boundary in boundaries:
        if boundary <= cursor:
            continue
        piece = segment[cursor:boundary]
        left = len(piece) - len(piece.lstrip())
        right = len(piece.rstrip())
        if right > left:
            spans.append((start + cursor + left, start + cursor + right))
        cursor = boundary
    return spans


def _bounded_token_units(
    tokenizer,
    text: str,
    start: int,
    end: int,
    hard_max_tokens: int,
    boundary: str,
) -> list[dict[str, Any]]:
    """Return source-aligned units no longer than ``hard_max_tokens``."""
    segment = text[start:end]
    offsets = token_offsets(tokenizer, segment)
    if not offsets:
        return []
    units: list[dict[str, Any]] = []
    for token_start in range(0, len(offsets), hard_max_tokens):
        window = offsets[token_start : token_start + hard_max_tokens]
        units.append(
            {
                "start": start + window[0][0],
                "end": start + window[-1][1],
                "tokens": len(window),
                "boundary": boundary if len(offsets) <= hard_max_tokens else "token-window",
            }
        )
    return units


def structural_units(tokenizer, text: str, hard_max_tokens: int) -> list[dict[str, Any]]:
    """Split at paragraph/line boundaries, then sentences, then token windows.

    The returned spans always point into the original Parent text. Short lines are
    kept intact. Long prose is split only at sentence boundaries where possible;
    a token window is the final safety fallback.
    """
    units: list[dict[str, Any]] = []
    for line_start, line_end in nonempty_line_spans(text):
        line_tokens = token_offsets(tokenizer, text[line_start:line_end])
        if len(line_tokens) <= hard_max_tokens:
            units.extend(
                _bounded_token_units(
                    tokenizer, text, line_start, line_end, hard_max_tokens, "line"
                )
            )
            continue

        for sentence_start, sentence_end in sentence_spans(text, line_start, line_end):
            units.extend(
                _bounded_token_units(
                    tokenizer,
                    text,
                    sentence_start,
                    sentence_end,
                    hard_max_tokens,
                    "sentence",
                )
            )
    return units


def split_parent_structured(
    tokenizer,
    text: str,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    overlap_max_tokens: int,
    overlap_ratio_cap: float,
) -> list[dict[str, Any]]:
    """Create variable-length Child chunks while respecting semantic boundaries."""
    units = structural_units(tokenizer, text, max_tokens)
    if not units:
        return []
    children: list[dict[str, Any]] = []
    start_index = 0
    while start_index < len(units):
        end_index = start_index
        token_total = 0
        while end_index < len(units):
            unit_tokens = int(units[end_index]["tokens"])
            proposed = token_total + unit_tokens
            if end_index > start_index and proposed > max_tokens:
                break
            if end_index > start_index and token_total >= min_tokens and proposed > target_tokens:
                if abs(token_total - target_tokens) <= abs(proposed - target_tokens):
                    break
            token_total = proposed
            end_index += 1
            if token_total >= target_tokens:
                break

        first, last = units[start_index], units[end_index - 1]
        local_start, local_end = int(first["start"]), int(last["end"])
        children.append(
            {
                "local_char_start": local_start,
                "local_char_end": local_end,
                "text": text[local_start:local_end],
                "token_count": token_total,
                "start_boundary": first["boundary"],
                "end_boundary": last["boundary"],
            }
        )
        if end_index >= len(units):
            break

        overlap_limit = min(
            overlap_max_tokens,
            max(0, int(target_tokens * overlap_ratio_cap)),
        )
        overlap_start = end_index
        overlap_total = 0
        for candidate in range(end_index - 1, start_index, -1):
            candidate_tokens = int(units[candidate]["tokens"])
            if overlap_total + candidate_tokens > overlap_limit:
                break
            overlap_total += candidate_tokens
            overlap_start = candidate
        start_index = overlap_start if overlap_start < end_index else end_index
    return children


def split_parent(
    tokenizer,
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]]:
    units = atomic_units(tokenizer, text, hard_max_tokens)
    if not units:
        return []
    children: list[dict[str, Any]] = []
    start_index = 0
    while start_index < len(units):
        end_index = start_index
        token_total = 0
        while end_index < len(units):
            unit_tokens = int(units[end_index]["tokens"])
            if end_index > start_index and token_total + unit_tokens > target_tokens:
                break
            token_total += unit_tokens
            end_index += 1
            if token_total >= target_tokens:
                break

        first = units[start_index]
        last = units[end_index - 1]
        local_start = int(first["start"])
        local_end = int(last["end"])
        child_text = text[local_start:local_end]
        children.append(
            {
                "local_char_start": local_start,
                "local_char_end": local_end,
                "text": child_text,
                "token_count": token_total,
            }
        )
        if end_index >= len(units):
            break
        if overlap_tokens <= 0:
            start_index = end_index
            continue

        overlap_start = end_index
        overlap_total = 0
        for candidate in range(end_index - 1, start_index, -1):
            candidate_tokens = int(units[candidate]["tokens"])
            if overlap_total + candidate_tokens > overlap_tokens:
                break
            overlap_total += candidate_tokens
            overlap_start = candidate
        start_index = overlap_start if overlap_start < end_index else end_index
    return children


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)] if ordered else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents-dir", type=Path, default=PROJECT_ROOT / "data/stage1/chunks")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=PROJECT_ROOT / "data/stage2/evidence",
    )
    parser.add_argument(
        "--index-config",
        type=Path,
        default=PROJECT_ROOT / "vector_db/stage1/minilm/raw/kb_config.json",
    )
    parser.add_argument("--target-tokens", type=int, default=200)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--hard-max-tokens", type=int, default=400)
    parser.add_argument(
        "--strategy", choices=("fixed", "structured"), default="fixed",
        help="fixed preserves the Stage-2 splitter; structured uses soft limits and semantic boundaries.",
    )
    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--overlap-ratio-cap", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if min(args.target_tokens, args.hard_max_tokens) <= 0:
        raise ValueError("Token limits must be positive")
    if not 0 <= args.overlap_tokens < args.target_tokens:
        raise ValueError("overlap_tokens must be in [0, target_tokens)")
    if args.target_tokens > args.hard_max_tokens:
        raise ValueError("target_tokens must not exceed hard_max_tokens")

    min_tokens = args.min_tokens or max(1, round(args.target_tokens * 0.65))
    max_tokens = args.max_tokens or args.hard_max_tokens
    if not 0 < min_tokens <= args.target_tokens <= max_tokens:
        raise ValueError("structured limits must satisfy 0 < min <= target <= max")
    if not 0 <= args.overlap_ratio_cap < 1:
        raise ValueError("overlap_ratio_cap must be in [0, 1)")

    variant = (
        f"minilm_c{args.target_tokens}_o{args.overlap_tokens}_v1"
        if args.strategy == "fixed"
        else (
            f"minilm_struct_t{args.target_tokens}_min{min_tokens}_max{max_tokens}"
            f"_ocap{args.overlap_tokens}_r{int(args.overlap_ratio_cap * 100)}_v1"
        )
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else PROJECT_ROOT / "data/stage2/parent_child" / variant
    )
    parents_output = output_dir / "parents"
    children_output = output_dir / "children"
    parents_output.mkdir(parents=True, exist_ok=True)
    children_output.mkdir(parents=True, exist_ok=True)

    index_config = json.loads(args.index_config.resolve().read_text(encoding="utf-8"))
    model_name = str(index_config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.offline)
    tokenizer.model_max_length = 1_000_000_000

    span_rows = load_jsonl(args.evidence_dir.resolve() / "legacy_chunk_spans.jsonl")
    span_by_parent = {str(row["chunk_id"]): row for row in span_rows}
    all_parent_ids: list[str] = []
    all_child_ids: set[str] = set()
    token_lengths: list[int] = []
    children_per_parent: list[int] = []
    per_file: dict[str, dict[str, int]] = {}

    chunk_files = discover_chunk_files(args.parents_dir.resolve())
    for filename in chunk_files:
        input_path = args.parents_dir.resolve() / filename
        parents = json.loads(input_path.read_text(encoding="utf-8"))
        parent_records: list[dict[str, Any]] = []
        child_records: list[dict[str, Any]] = []
        for parent in parents:
            parent_id = str(parent["id"])
            if parent_id not in span_by_parent:
                raise KeyError(f"Missing evidence span for parent: {parent_id}")
            span = span_by_parent[parent_id]
            text = str(parent.get("text") or "")
            if args.strategy == "structured":
                pieces = split_parent_structured(
                    tokenizer,
                    text,
                    args.target_tokens,
                    min_tokens,
                    max_tokens,
                    args.overlap_tokens,
                    args.overlap_ratio_cap,
                )
            else:
                pieces = split_parent(
                    tokenizer,
                    text,
                    args.target_tokens,
                    args.overlap_tokens,
                    args.hard_max_tokens,
                )
            if not pieces:
                raise ValueError(f"Parent generated no children: {parent_id}")

            parent_record = dict(parent)
            parent_record.update(
                {
                    "parent_id": parent_id,
                    "chunk_type": "parent",
                    "parent_child_schema": SCHEMA_VERSION,
                    "source_id": span["source_id"],
                    "source_char_start": span["char_start"],
                    "source_char_end": span["char_end"],
                }
            )
            parent_records.append(parent_record)
            all_parent_ids.append(parent_id)
            children_per_parent.append(len(pieces))

            for position, piece in enumerate(pieces, 1):
                child_id = f"{parent_id}__child_{position:03d}"
                if child_id in all_child_ids:
                    raise ValueError(f"Duplicate child ID: {child_id}")
                all_child_ids.add(child_id)
                source_start = int(span["char_start"]) + int(piece["local_char_start"])
                source_end = int(span["char_start"]) + int(piece["local_char_end"])
                child_text = str(piece["text"])
                token_count = int(piece["token_count"])
                token_lengths.append(token_count)
                child_records.append(
                    {
                        "id": child_id,
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "position": position,
                        "chunk_type": "child",
                        "parent_child_schema": SCHEMA_VERSION,
                        "text": child_text,
                        "text_sha256": sha256_text(child_text),
                        "token_count": token_count,
                        "source_id": span["source_id"],
                        "source_char_start": source_start,
                        "source_char_end": source_end,
                        "parent_char_start": span["char_start"],
                        "parent_char_end": span["char_end"],
                        "local_char_start": piece["local_char_start"],
                        "local_char_end": piece["local_char_end"],
                        "source": parent.get("source"),
                        "subject": parent.get("subject"),
                        "lang": parent.get("lang"),
                        "course": parent.get("course") or parent.get("subject"),
                        "document_title": parent.get("document_title"),
                        "chapter_path": parent.get("chapter_path"),
                        "chapter_titles": parent.get("chapter_titles", []),
                        "page_start": parent.get("page_start"),
                        "page_end": parent.get("page_end"),
                        "document_type": parent.get("document_type"),
                        "language": parent.get("language") or parent.get("lang"),
                    }
                )

        (parents_output / filename).write_text(
            json.dumps(parent_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (children_output / filename).write_text(
            json.dumps(child_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        per_file[filename] = {"parents": len(parent_records), "children": len(child_records)}

    if len(set(all_parent_ids)) != len(all_parent_ids):
        raise ValueError("Parent IDs are not unique")

    config = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "embedding_model": model_name,
        "parent_strategy": "preserve-stage1-chunk",
        "child_strategy": (
            "structure-aware-variable-length" if args.strategy == "structured"
            else "line-boundary-token-budget"
        ),
        "target_tokens": args.target_tokens,
        "overlap_tokens": args.overlap_tokens,
        "hard_max_tokens": max_tokens if args.strategy == "structured" else args.hard_max_tokens,
        "min_tokens": min_tokens if args.strategy == "structured" else None,
        "overlap_ratio_cap": args.overlap_ratio_cap if args.strategy == "structured" else None,
        "parent_id_compatibility": "Stage-1 IDs preserved exactly",
        "child_id_format": "<parent_id>__child_<001-based-position>",
        "parents_dir": str(parents_output),
        "children_dir": str(children_output),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "variant": variant,
        "parents": len(all_parent_ids),
        "children": len(all_child_ids),
        "parent_ids_preserved": True,
        "per_file": per_file,
        "children_per_parent": {
            "min": min(children_per_parent),
            "median": percentile(children_per_parent, 0.5),
            "p95": percentile(children_per_parent, 0.95),
            "max": max(children_per_parent),
        },
        "child_tokens": {
            "min": min(token_lengths),
            "p25": percentile(token_lengths, 0.25),
            "median": percentile(token_lengths, 0.5),
            "p75": percentile(token_lengths, 0.75),
            "p95": percentile(token_lengths, 0.95),
            "max": max(token_lengths),
        },
        "course_children": dict(Counter(
            child.get("course")
            for filename in chunk_files
            for child in json.loads((children_output / filename).read_text(encoding="utf-8"))
        )),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
