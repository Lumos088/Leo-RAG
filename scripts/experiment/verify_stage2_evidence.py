"""Verify the Stage-2 evidence baseline before Parent-Child experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SCHEMA = "stage2-evidence-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=PROJECT_ROOT / "data/stage2/evidence",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=PROJECT_ROOT / "data/stage1/chunks",
    )
    args = parser.parse_args()

    evidence_dir = args.evidence_dir.resolve()
    sources = load_jsonl(evidence_dir / "sources.jsonl")
    spans = load_jsonl(evidence_dir / "legacy_chunk_spans.jsonl")
    evidence_units = load_jsonl(evidence_dir / "evidence_units.jsonl")
    schema = json.loads((evidence_dir / "schema.json").read_text(encoding="utf-8"))
    migration = json.loads(
        (evidence_dir / "migration_report.json").read_text(encoding="utf-8")
    )

    chunks: list[dict] = []
    for path in sorted(args.chunks_dir.resolve().glob("*.json")):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    chunk_by_id = {str(chunk["id"]): chunk for chunk in chunks}
    source_by_id = {str(source["source_id"]): source for source in sources}
    evidence_by_id = {str(unit["evidence_id"]): unit for unit in evidence_units}

    failures: list[str] = []
    if schema.get("schema_version") != EXPECTED_SCHEMA:
        failures.append("Unexpected schema version")
    if len(chunk_by_id) != len(chunks):
        failures.append("Stage-1 chunk IDs are not unique")
    if len(source_by_id) != len(sources):
        failures.append("Source IDs are not unique")
    if len(evidence_by_id) != len(evidence_units):
        failures.append("Evidence IDs are not unique")

    canonical_by_source: dict[str, str] = {}
    for source_id, source in source_by_id.items():
        canonical_path = evidence_dir / "canonical_sources" / f"{source_id}.txt"
        if not canonical_path.exists():
            failures.append(f"Missing canonical source: {source_id}")
            continue
        canonical = canonical_path.read_text(encoding="utf-8")
        canonical_by_source[source_id] = canonical
        if len(canonical) != int(source["canonical_char_length"]):
            failures.append(f"Canonical length mismatch: {source_id}")
        if sha256_text(canonical) != source["canonical_sha256"]:
            failures.append(f"Canonical hash mismatch: {source_id}")

    span_ids: set[str] = set()
    for span in spans:
        chunk_id = str(span["chunk_id"])
        if chunk_id in span_ids:
            failures.append(f"Duplicate chunk span: {chunk_id}")
            continue
        span_ids.add(chunk_id)
        source_id = str(span["source_id"])
        canonical = canonical_by_source.get(source_id)
        chunk = chunk_by_id.get(chunk_id)
        if canonical is None or chunk is None:
            failures.append(f"Unresolved span: {chunk_id}")
            continue
        start = int(span["char_start"])
        end = int(span["char_end"])
        if not (0 <= start < end <= len(canonical)):
            failures.append(f"Invalid character interval: {chunk_id}")
            continue
        recovered = canonical[start:end]
        if recovered != str(chunk.get("text") or "").strip():
            failures.append(f"Recovered text mismatch: {chunk_id}")
        if sha256_text(recovered) != span["text_sha256"]:
            failures.append(f"Span hash mismatch: {chunk_id}")

    qrels_path = evidence_dir / "evidence_qrels.csv"
    with qrels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        qrels = list(csv.DictReader(handle))
    positive_qids: set[str] = set()
    for row in qrels:
        evidence_id = str(row["evidence_id"])
        if evidence_id not in evidence_by_id:
            failures.append(f"Unknown evidence ID in qrels: {evidence_id}")
        if str(row["legacy_chunk_id"]) not in chunk_by_id:
            failures.append(f"Unknown legacy chunk in qrels: {row['legacy_chunk_id']}")
        if int(row["rel"]) > 0:
            positive_qids.add(str(row["qid"]))

    report = {
        "ok": not failures,
        "failures": failures[:25],
        "schema_version": schema.get("schema_version"),
        "sources": len(sources),
        "legacy_chunks": len(chunks),
        "verified_spans": len(span_ids),
        "evidence_units": len(evidence_units),
        "judged_rows": len(qrels),
        "positive_qids": len(positive_qids),
        "migration_report_consistent": (
            migration.get("located_legacy_chunks") == len(span_ids)
            and migration.get("evidence_units") == len(evidence_units)
            and migration.get("judged_rows") == len(qrels)
        ),
    }
    if not report["migration_report_consistent"]:
        report["ok"] = False
        report["failures"].append("Migration report counts are inconsistent")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
