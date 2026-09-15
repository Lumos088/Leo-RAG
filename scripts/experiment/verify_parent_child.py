"""Verify Parent-Child IDs, spans, text recovery, and parent coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CHUNK_FILES = ("ds_chunks.json", "os_chunks.json", "cn_chunks.json")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_files(directory: Path, *, include_all: bool = False) -> list[dict]:
    records: list[dict] = []
    filenames = (
        sorted(path.name for path in directory.glob("*_chunks.json"))
        if include_all
        else LEGACY_CHUNK_FILES
    )
    for filename in filenames:
        records.extend(json.loads((directory / filename).read_text(encoding="utf-8")))
    return records


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", type=Path, default=PROJECT_ROOT / "data/stage1/chunks")
    parser.add_argument(
        "--parent-child-dir",
        type=Path,
        default=PROJECT_ROOT / "data/stage2/parent_child/minilm_c200_o50_v1",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=PROJECT_ROOT / "data/stage2/evidence",
    )
    args = parser.parse_args()

    stage1 = load_files(args.stage1_dir.resolve())
    parents = load_files(args.parent_child_dir.resolve() / "parents", include_all=True)
    children = load_files(args.parent_child_dir.resolve() / "children", include_all=True)
    config = json.loads(
        (args.parent_child_dir.resolve() / "config.json").read_text(encoding="utf-8")
    )
    sources = load_jsonl(args.evidence_dir.resolve() / "sources.jsonl")

    stage1_by_id = {str(row["id"]): row for row in stage1}
    parent_by_id = {str(row["id"]): row for row in parents}
    child_by_id = {str(row["id"]): row for row in children}
    children_by_parent: dict[str, list[dict]] = defaultdict(list)
    for child in children:
        children_by_parent[str(child["parent_id"])].append(child)

    failures: list[str] = []
    if not set(stage1_by_id).issubset(parent_by_id):
        failures.append("Stage-1 Parent IDs are not preserved")
    if len(child_by_id) != len(children):
        failures.append("Child IDs are not unique")

    canonical: dict[str, str] = {}
    for source in sources:
        source_id = str(source["source_id"])
        path = args.evidence_dir.resolve() / "canonical_sources" / f"{source_id}.txt"
        canonical[source_id] = path.read_text(encoding="utf-8")

    max_tokens = 0
    parents_without_children = 0
    for parent_id, parent in parent_by_id.items():
        original = stage1_by_id.get(parent_id)
        if original is None:
            continue
        if str(parent.get("text") or "") != str(original.get("text") or ""):
            failures.append(f"Parent text changed: {parent_id}")
        parent_children = sorted(
            children_by_parent.get(parent_id, []), key=lambda row: int(row["position"])
        )
        if not parent_children:
            parents_without_children += 1
            continue
        expected_positions = list(range(1, len(parent_children) + 1))
        positions = [int(row["position"]) for row in parent_children]
        if positions != expected_positions:
            failures.append(f"Child positions are not contiguous: {parent_id}")

        intervals: list[tuple[int, int]] = []
        parent_text = str(parent["text"])
        for child in parent_children:
            position = int(child["position"])
            expected_id = f"{parent_id}__child_{position:03d}"
            if child["id"] != expected_id or child["child_id"] != expected_id:
                failures.append(f"Invalid child ID: {child['id']}")
            local_start = int(child["local_char_start"])
            local_end = int(child["local_char_end"])
            if not (0 <= local_start < local_end <= len(parent_text)):
                failures.append(f"Invalid local span: {child['id']}")
                continue
            recovered = parent_text[local_start:local_end]
            if recovered != child["text"]:
                failures.append(f"Child text mismatch: {child['id']}")
            if sha256_text(recovered) != child["text_sha256"]:
                failures.append(f"Child hash mismatch: {child['id']}")
            source_text = canonical.get(str(child["source_id"]))
            source_start = int(child["source_char_start"])
            source_end = int(child["source_char_end"])
            if source_text is None or source_text[source_start:source_end] != recovered:
                failures.append(f"Canonical source mismatch: {child['id']}")
            intervals.append((local_start, local_end))
            max_tokens = max(max_tokens, int(child["token_count"]))

        covered = [False] * len(parent_text)
        for start, end in intervals:
            covered[start:end] = [True] * (end - start)
        if any(not hit and not char.isspace() for hit, char in zip(covered, parent_text)):
            failures.append(f"Child spans do not cover parent content: {parent_id}")

    if parents_without_children:
        failures.append(f"Parents without children: {parents_without_children}")
    hard_max = int(config.get("hard_max_tokens", config.get("child_max_tokens", 0)))
    if hard_max <= 0:
        failures.append("Missing positive hard_max_tokens/child_max_tokens in config")
    if max_tokens > hard_max:
        failures.append(f"Child token limit exceeded: {max_tokens}>{hard_max}")

    report = {
        "ok": not failures,
        "failures": failures[:25],
        "variant": config.get("variant"),
        "stage1_parents": len(stage1),
        "verified_parents": len(parents),
        "verified_children": len(children),
        "parent_ids_preserved": set(stage1_by_id).issubset(parent_by_id),
        "additional_parents": len(set(parent_by_id) - set(stage1_by_id)),
        "parents_without_children": parents_without_children,
        "max_child_tokens": max_tokens,
        "hard_max_tokens": hard_max,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
