"""Verify Stage-1 artifacts and print actionable failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment.evaluate_retrieval import load_qrels


CHUNK_FILES = ("ds_chunks.json", "os_chunks.json", "cn_chunks.json")
REQUIRED_METADATA = {
    "course",
    "document_title",
    "chapter_path",
    "chapter_titles",
    "document_type",
    "language",
    "metadata_version",
}


def id_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def read_index(path: Path) -> faiss.Index:
    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype="uint8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, default=PROJECT_ROOT / "data" / "chunks")
    parser.add_argument("--enriched-dir", type=Path, default=PROJECT_ROOT / "data" / "stage1" / "chunks")
    parser.add_argument("--vector-root", type=Path, default=PROJECT_ROOT / "vector_db" / "stage1")
    parser.add_argument("--qrels", type=Path, default=PROJECT_ROOT / "qrels.csv")
    args = parser.parse_args()

    original: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for filename in CHUNK_FILES:
        original.extend(json.loads((args.original_dir / filename).read_text(encoding="utf-8")))
        enriched.extend(json.loads((args.enriched_dir / filename).read_text(encoding="utf-8")))

    failures: list[str] = []
    original_ids = [str(item.get("id")) for item in original]
    enriched_ids = [str(item.get("id")) for item in enriched]
    if original_ids != enriched_ids:
        failures.append("Chunk ID order differs from production chunks")
    original_text = {str(item["id"]): str(item.get("text") or "") for item in original}
    changed_text = [
        str(item["id"])
        for item in enriched
        if original_text.get(str(item["id"])) != str(item.get("text") or "")
    ]
    if changed_text:
        failures.append(f"Chunk text changed for {len(changed_text)} IDs")
    missing_fields = sum(not REQUIRED_METADATA.issubset(item) for item in enriched)
    if missing_fields:
        failures.append(f"Required metadata fields missing from {missing_fields} chunks")
    empty_course = sum(not item.get("course") for item in enriched)
    empty_document = sum(not item.get("document_title") for item in enriched)
    chapter_coverage = sum(bool(item.get("chapter_path")) for item in enriched) / max(len(enriched), 1)
    if empty_course or empty_document:
        failures.append(f"Empty course={empty_course}, empty document_title={empty_document}")

    indexes = []
    if args.vector_root.exists():
        for config_path in sorted(args.vector_root.glob("*/*/kb_config.json")):
            directory = config_path.parent
            metadata = json.loads((directory / "kb_meta.json").read_text(encoding="utf-8"))
            index = read_index(directory / "kb.index")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            valid = index.ntotal == len(metadata) == len(enriched) and index.d == int(config["dim"])
            if not valid:
                failures.append(f"Index/config/meta mismatch: {directory}")
            indexes.append(
                {
                    "name": str(directory.relative_to(args.vector_root)),
                    "vectors": int(index.ntotal),
                    "dimension": int(index.d),
                    "valid": valid,
                }
            )

    qrels, _ = load_qrels(args.qrels)
    positive_qids = [qid for qid, values in qrels.items() if any(rel > 0 for rel in values.values())]
    report = {
        "ok": not failures,
        "failures": failures,
        "chunks": len(enriched),
        "chunk_id_sha256": id_digest(enriched_ids),
        "chunk_ids_preserved": original_ids == enriched_ids,
        "chunk_text_preserved": not changed_text,
        "chapter_coverage": chapter_coverage,
        "indexes": indexes,
        "qrels_positive_queries": len(positive_qids),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
