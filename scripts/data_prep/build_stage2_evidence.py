"""Build a stable evidence-level evaluation baseline for Parent-Child chunking.

The legacy qrels point at Stage-1 chunk IDs.  This script reconstructs every
legacy chunk's character span in a versioned canonical source text, then maps
non-empty relevance judgements to evidence IDs derived from source + span +
text hash.  Future chunking strategies can therefore be evaluated against the
same evidence spans without reusing their own chunk IDs as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNK_FILES = ("ds_chunks.json", "os_chunks.json", "cn_chunks.json")
CLEANED_FILES = (
    "ds_cleaned_documents.json",
    "os_cleaned_documents.json",
    "cn_cleaned_documents.json",
)
SCHEMA_VERSION = "stage2-evidence-v1"
CANONICALIZATION_VERSION = "stage1-lines-v1"
SINGLE_SENT_MAX = 400


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def detect_encoding(path: Path) -> str:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            payload.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unable to decode {path}")


def split_long_sentence(value: str, max_len: int = SINGLE_SENT_MAX) -> list[str]:
    value = value.strip()
    if len(value) <= max_len:
        return [value]
    parts = re.split(r"([。！？!?\.])", value)
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        if len(buffer) + len(part) <= max_len:
            buffer += part
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = part
    if buffer:
        chunks.append(buffer.strip())
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_len:
            result.append(chunk)
        else:
            result.extend(
                piece
                for start in range(0, len(chunk), max_len)
                if (piece := chunk[start : start + max_len].strip())
            )
    return result


def canonicalize(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if len(line) > SINGLE_SENT_MAX:
            lines.extend(split_long_sentence(line))
        else:
            lines.append(line)
    return "\n".join(lines)


def load_json_lists(directory: Path, filenames: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename in filenames:
        path = directory / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list: {path}")
        records.extend(payload)
    return records


def make_source_id(source: str) -> str:
    return "src_" + sha256_text(source.strip())[:16]


def make_evidence_id(source_id: str, start: int, end: int, text_hash: str) -> str:
    identity = f"{source_id}|{start}|{end}|{text_hash}"
    return "ev_" + sha256_text(identity)[:20]


def locate_chunks(
    chunks: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk["source"])].append(chunk)

    spans: list[dict[str, Any]] = []
    span_by_chunk: dict[str, dict[str, Any]] = {}
    canonical_sources: dict[str, str] = {}

    for source, source_chunks in grouped.items():
        if source not in documents:
            raise KeyError(f"No cleaned document for source: {source}")
        document = documents[source]
        canonical = canonicalize(str(document.get("text") or ""))
        source_id = make_source_id(source)
        canonical_sources[source_id] = canonical
        previous_start = 0

        for chunk in source_chunks:
            chunk_id = str(chunk["id"])
            chunk_text = str(chunk.get("text") or "").strip()
            start = canonical.find(chunk_text, previous_start)
            if start < 0:
                # A repeated passage can make the monotonic hint too strict.
                start = canonical.find(chunk_text)
            if start < 0:
                raise ValueError(f"Chunk text not found in canonical source: {chunk_id}")
            end = start + len(chunk_text)
            if canonical[start:end] != chunk_text:
                raise AssertionError(f"Span verification failed: {chunk_id}")
            previous_start = start + 1
            record = {
                "schema_version": SCHEMA_VERSION,
                "canonicalization_version": CANONICALIZATION_VERSION,
                "chunking_version": "stage1-legacy-v1",
                "chunk_id": chunk_id,
                "source_id": source_id,
                "source": source,
                "course": chunk.get("course") or chunk.get("subject"),
                "document_title": chunk.get("document_title"),
                "chapter_path": chunk.get("chapter_path"),
                "char_start": start,
                "char_end": end,
                "char_length": end - start,
                "text_sha256": sha256_text(chunk_text),
            }
            spans.append(record)
            span_by_chunk[chunk_id] = record

    return spans, span_by_chunk, canonical_sources


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=Path, default=PROJECT_ROOT / "data/stage1/chunks")
    parser.add_argument("--cleaned-dir", type=Path, default=PROJECT_ROOT / "data/cleaned")
    parser.add_argument("--qrels", type=Path, default=PROJECT_ROOT / "qrels.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/stage2/evidence",
    )
    args = parser.parse_args()

    chunks = load_json_lists(args.chunks_dir.resolve(), CHUNK_FILES)
    chunk_by_id = {str(chunk["id"]): chunk for chunk in chunks}
    if len(chunk_by_id) != len(chunks):
        raise ValueError("Stage-1 chunk IDs are not unique")
    cleaned = load_json_lists(args.cleaned_dir.resolve(), CLEANED_FILES)
    documents = {str(document["source"]): document for document in cleaned}
    if len(documents) != len(cleaned):
        raise ValueError("Cleaned document sources are not unique")

    spans, span_by_chunk, canonical_sources = locate_chunks(chunks, documents)
    if len(span_by_chunk) != len(chunks):
        raise AssertionError("Chunk IDs are not unique")

    qrels_path = args.qrels.resolve()
    with qrels_path.open("r", encoding=detect_encoding(qrels_path), newline="") as handle:
        qrel_rows = list(csv.DictReader(handle))

    evidence_by_id: dict[str, dict[str, Any]] = {}
    evidence_qrels: list[dict[str, Any]] = []
    blank_rows = 0
    positive_rows = 0
    zero_rows = 0

    for row in qrel_rows:
        rel_text = str(row.get("rel") or "").strip()
        if not rel_text:
            blank_rows += 1
            continue
        rel = int(float(rel_text))
        chunk_id = str(row.get("doc_id") or "").strip()
        if chunk_id not in span_by_chunk:
            raise KeyError(f"qrels chunk is missing from Stage-1 corpus: {chunk_id}")
        span = span_by_chunk[chunk_id]
        chunk_text = str(chunk_by_id[chunk_id]["text"])
        evidence_id = make_evidence_id(
            str(span["source_id"]),
            int(span["char_start"]),
            int(span["char_end"]),
            str(span["text_sha256"]),
        )
        evidence_by_id.setdefault(
            evidence_id,
            {
                "schema_version": SCHEMA_VERSION,
                "evidence_id": evidence_id,
                "source_id": span["source_id"],
                "source": span["source"],
                "course": span["course"],
                "document_title": span["document_title"],
                "chapter_path": span["chapter_path"],
                "char_start": span["char_start"],
                "char_end": span["char_end"],
                "char_length": span["char_length"],
                "text_sha256": span["text_sha256"],
                "text": chunk_text,
                "legacy_chunk_id": chunk_id,
            },
        )
        evidence_qrels.append(
            {
                "qid": str(row.get("qid") or "").strip(),
                "evidence_id": evidence_id,
                "rel": rel,
                "legacy_chunk_id": chunk_id,
                "annotation_source": str(row.get("source") or "").strip(),
            }
        )
        if rel > 0:
            positive_rows += 1
        else:
            zero_rows += 1

    output_dir = args.output_dir.resolve()
    canonical_dir = output_dir / "canonical_sources"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    source_manifest: list[dict[str, Any]] = []
    for source, document in sorted(documents.items()):
        source_id = make_source_id(source)
        canonical = canonical_sources[source_id]
        canonical_path = canonical_dir / f"{source_id}.txt"
        canonical_path.write_text(canonical, encoding="utf-8", newline="\n")
        source_manifest.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_id": source_id,
                "source": source,
                "document_id": document.get("id"),
                "course": document.get("subject"),
                "language": document.get("lang"),
                "canonicalization_version": CANONICALIZATION_VERSION,
                "canonical_char_length": len(canonical),
                "canonical_sha256": sha256_text(canonical),
                "canonical_path": str(canonical_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                if canonical_path.is_relative_to(PROJECT_ROOT)
                else str(canonical_path),
            }
        )

    write_jsonl(output_dir / "sources.jsonl", source_manifest)
    write_jsonl(output_dir / "legacy_chunk_spans.jsonl", spans)
    write_jsonl(output_dir / "evidence_units.jsonl", evidence_by_id.values())

    qrels_output = output_dir / "evidence_qrels.csv"
    with qrels_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "qid",
                "evidence_id",
                "rel",
                "legacy_chunk_id",
                "annotation_source",
            ],
        )
        writer.writeheader()
        writer.writerows(evidence_qrels)

    schema = {
        "schema_version": SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "identity": "evidence_id = SHA256(source_id|char_start|char_end|text_sha256)[:20]",
        "span_semantics": "Zero-based half-open [char_start, char_end) in canonical source text.",
        "unjudged_semantics": "Blank legacy rel values remain unjudged and are not converted to rel=0.",
        "future_chunk_mapping": {
            "same_source_required": True,
            "intersection_chars": "max(0, min(chunk_end,evidence_end)-max(chunk_start,evidence_start))",
            "evidence_coverage": "intersection_chars / evidence_char_length",
            "chunk_purity": "intersection_chars / chunk_char_length",
            "initial_match_rule": "evidence_coverage >= 0.50 OR chunk_purity >= 0.80",
            "note": "Report both coverage values; validate thresholds on a reviewed sample before final evaluation.",
        },
    }
    (output_dir / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    qids_with_positive = {
        row["qid"] for row in evidence_qrels if int(row["rel"]) > 0
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "sources": len(source_manifest),
        "legacy_chunks": len(chunks),
        "located_legacy_chunks": len(spans),
        "qrels_rows": len(qrel_rows),
        "judged_rows": len(evidence_qrels),
        "blank_unjudged_rows": blank_rows,
        "positive_rows": positive_rows,
        "zero_rows": zero_rows,
        "positive_qids": len(qids_with_positive),
        "evidence_units": len(evidence_by_id),
        "relevance_distribution": dict(Counter(str(row["rel"]) for row in evidence_qrels)),
        "chunk_span_digest": sha256_text(
            "\n".join(
                f"{row['chunk_id']}|{row['source_id']}|{row['char_start']}|{row['char_end']}|{row['text_sha256']}"
                for row in spans
            )
        ),
    }
    (output_dir / "migration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
