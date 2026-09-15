"""Freeze a lightweight, reproducible Stage 1 retrieval baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from evaluate_retrieval import detect_text_encoding, evaluate_directory, load_qrels


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "stage1" / "baseline"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_topics(path: Path) -> list[dict[str, str]]:
    encoding = detect_text_encoding(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def load_chunk_ids(paths: list[Path]) -> tuple[list[str], dict[str, int]]:
    ids: list[str] = []
    counts: dict[str, int] = {}
    for path in paths:
        chunks = json.loads(path.read_text(encoding="utf-8"))
        file_ids = [str(chunk["id"]) for chunk in chunks]
        if len(file_ids) != len(set(file_ids)):
            raise ValueError(f"Duplicate chunk IDs in {path}")
        counts[path.name] = len(file_ids)
        ids.extend(file_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs across chunk files")
    return ids, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = [
        PROJECT_ROOT / "data" / "chunks" / name
        for name in ("ds_chunks.json", "os_chunks.json", "cn_chunks.json")
    ]
    tracked_paths = chunk_paths + [
        PROJECT_ROOT / "vector_db" / "kb.index",
        PROJECT_ROOT / "vector_db" / "kb_meta.json",
        PROJECT_ROOT / "vector_db" / "kb_config.json",
        PROJECT_ROOT / "qrels.csv",
        PROJECT_ROOT / "topics.csv",
    ]
    missing = [str(path) for path in tracked_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing baseline files: {missing}")

    chunk_ids, chunk_counts = load_chunk_ids(chunk_paths)
    chunk_id_digest = hashlib.sha256("\n".join(sorted(chunk_ids)).encode("utf-8")).hexdigest()
    (output_dir / "chunk_ids.txt").write_text("\n".join(sorted(chunk_ids)) + "\n", encoding="utf-8")

    qrels, _ = load_qrels(PROJECT_ROOT / "qrels.csv")
    positive_qids = sorted(
        qid for qid, values in qrels.items() if any(rel > 0 for rel in values.values())
    )
    topics = load_topics(PROJECT_ROOT / "topics.csv")
    topic_qids = sorted({(row.get("qid") or "").strip() for row in topics if row.get("qid")})

    evaluation = evaluate_directory(PROJECT_ROOT / "qrels.csv", PROJECT_ROOT / "runs")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    config = json.loads((PROJECT_ROOT / "vector_db" / "kb_config.json").read_text(encoding="utf-8"))
    manifest = {
        "baseline_version": "stage1-baseline-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "chunk_count": len(chunk_ids),
        "chunk_counts": chunk_counts,
        "chunk_id_sha256": chunk_id_digest,
        "vector_config": config,
        "topics": {
            "count": len(topic_qids),
            "qids": topic_qids,
        },
        "qrels": {
            "judged_query_count": len(qrels),
            "positive_query_count": len(positive_qids),
            "positive_qids": positive_qids,
            "queries_without_positive_judgements": sorted(set(topic_qids) - set(positive_qids)),
        },
        "files": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in tracked_paths
        },
        "notes": [
            "This snapshot stores hashes and IDs instead of copying the large FAISS index.",
            "Model comparisons must use the same positive-qid set until qrels coverage is expanded.",
            "Absolute paths inside the legacy kb_config are recorded as-is but are not trusted for loading.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(PROJECT_ROOT / "vector_db" / "kb_config.json", output_dir / "kb_config.original.json")

    print(f"baseline: {output_dir}")
    print(f"chunks: {len(chunk_ids)} id_sha256={chunk_id_digest}")
    print(
        f"topics: {len(topic_qids)} | qrels with positive judgements: {len(positive_qids)} | "
        f"missing positive labels: {len(set(topic_qids) - set(positive_qids))}"
    )


if __name__ == "__main__":
    main()
