"""Build the Stage 4 AI corpus without mutating any Stage 2 artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_prep.build_parent_child_chunks import split_parent, split_parent_structured
from scripts.pipeline.corpus_files import discover_chunk_files


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def download(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "RAG-stage4-builder/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def clean_markdown(raw: str) -> str:
    """Keep explanatory prose/headings while removing MDX plumbing and code."""
    raw = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", raw, flags=re.S)
    raw = re.sub(r"```.*?```", "\n", raw, flags=re.S)
    raw = re.sub(r"<Tip.*?</Tip>|<Warning.*?</Warning>", "\n", raw, flags=re.S)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"^\s*(import|export)\s+.*$", "", raw, flags=re.M)
    raw = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", raw)
    raw = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", raw)
    raw = re.sub(r"\{\{<.*?>\}\}", "", raw)
    raw = re.sub(r"\{#[^}]+\}", "", raw)
    raw = re.sub(r"^\s*:[^\n]+$", "", raw, flags=re.M)
    raw = re.sub(r"^\s*<!--.*?-->\s*$", "", raw, flags=re.M | re.S)
    raw = re.sub(r"^\s*\|?\s*:?-{3,}.*$", "", raw, flags=re.M)
    raw = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", raw)
    return html.unescape(raw).strip()


def copy_legacy(source: Path, target: Path) -> dict[str, str]:
    target.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for filename in discover_chunk_files(source):
        src = source / filename
        dst = target / filename
        shutil.copy2(src, dst)
        source_hash = hashlib.sha256(src.read_bytes()).hexdigest()
        if hashlib.sha256(dst.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError(f"Copy verification failed: {filename}")
        hashes[filename] = source_hash
    return hashes


def record_base(document: dict[str, Any], source: dict[str, Any], source_url: str) -> dict[str, Any]:
    topics = [str(item) for item in document["topics_zh"]]
    return {
        "source": source_url,
        "source_url": source_url,
        "source_repository": source["repository"],
        "source_revision": source["revision"],
        "source_slug": document["slug"],
        "source_collection": source["collection"],
        "source_license": source["license"],
        "source_license_url": source["license_url"],
        "source_attribution": source["attribution"],
        "acquired_at": "2026-09-14",
        "subject": "Artificial Intelligence",
        "course": "Artificial Intelligence",
        "document_title": document["title"],
        "document_type": "open_course_material",
        "lang": "en",
        "language": "en",
        "topics": topics,
        "metadata_version": "stage4-v1",
        "retrieval_context": "Course: Artificial Intelligence; Document: "
        + document["title"]
        + "; Chinese topics: "
        + ", ".join(topics),
    }


def choose_qrels(questions: list[dict[str, Any]], parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for parent in parents:
        by_source[str(parent["source_slug"])].append(parent)
    qrels: list[dict[str, Any]] = []
    for question in questions:
        candidates = by_source[question["source_slug"]]
        scored = []
        for parent in candidates:
            lowered = str(parent["text"]).lower()
            score = sum(term.lower() in lowered for term in question["answer_terms"])
            scored.append((score, parent))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        selected = [item for item in scored if item[0] > 0][:2] or scored[:1]
        for index, (score, parent) in enumerate(selected):
            qrels.append(
                {
                    "qid": question["qid"],
                    "doc_id": parent["id"],
                    "rel": 3 if index == 0 else 2,
                    "judgment_basis": "answer_terms_in_pinned_source",
                    "matched_terms": score,
                    "source_slug": question["source_slug"],
                }
            )
    return qrels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "config/stage4_sources.json")
    parser.add_argument("--eval-manifest", type=Path, default=PROJECT_ROOT / "config/stage4_eval_questions.json")
    parser.add_argument("--stage2-dir", type=Path, default=PROJECT_ROOT / "data/stage2/parent_child/minilm_c200_o50_v1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/stage4/parent_child/minilm_c200_o50_v1")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/stage4/sources")
    parser.add_argument("--model-name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--child-strategy", choices=("fixed", "structured"), default="fixed")
    parser.add_argument("--child-target-tokens", type=int, default=200)
    parser.add_argument("--child-min-tokens", type=int, default=30)
    parser.add_argument("--child-max-tokens", type=int, default=400)
    parser.add_argument("--child-overlap-max-tokens", type=int, default=50)
    parser.add_argument("--child-overlap-ratio-cap", type=float, default=0.25)
    parser.add_argument(
        "--reuse-raw", action="store_true",
        help="Use the already pinned source and license files instead of downloading them again.",
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    eval_manifest = json.loads(args.eval_manifest.resolve().read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    legacy_dir = args.stage2_dir.resolve()
    legacy_config_path = legacy_dir / "config.json"
    legacy_config = (
        json.loads(legacy_config_path.read_text(encoding="utf-8"))
        if legacy_config_path.is_file()
        else {}
    )
    parents_dir = output_dir / "parents"
    children_dir = output_dir / "children"
    raw_dir = args.raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    legacy_parent_hashes = copy_legacy(legacy_dir / "parents", parents_dir)
    legacy_child_hashes = copy_legacy(legacy_dir / "children", children_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.offline)
    tokenizer.model_max_length = 1_000_000_000
    ai_parents: list[dict[str, Any]] = []
    ai_children: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []

    for source in manifest["sources"]:
        license_raw_url = (
            f"https://raw.githubusercontent.com/{source['repository']}/"
            f"{source['revision']}/{source.get('license_path', 'LICENSE')}"
        )
        license_dir = raw_dir / "licenses"
        license_dir.mkdir(parents=True, exist_ok=True)
        license_file = license_dir / (source["repository"].replace("/", "__") + ".txt")
        if args.reuse_raw:
            if not license_file.is_file():
                raise FileNotFoundError(license_file)
        else:
            license_file.write_text(download(license_raw_url), encoding="utf-8")
        license_record = {
            "source_license_file": str(license_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source_license_sha256": hashlib.sha256(license_file.read_bytes()).hexdigest(),
        }
        for document in source["documents"]:
            source_url = (
                f"https://raw.githubusercontent.com/{source['repository']}/"
                f"{source['revision']}/{document['path']}"
            )
            raw_path = raw_dir / f"{document['slug']}.md"
            if args.reuse_raw:
                if not raw_path.is_file():
                    raise FileNotFoundError(raw_path)
                raw = raw_path.read_text(encoding="utf-8")
            else:
                raw = download(source_url)
            cleaned = clean_markdown(raw)
            if len(cleaned) < 800:
                raise ValueError(f"Source is unexpectedly short after cleaning: {document['slug']}")
            if not args.reuse_raw:
                raw_path.write_text(raw, encoding="utf-8")
            source_id = "ai_src_" + sha256_text(source_url)[:16]
            base = record_base(document, source, source_url)
            parent_spans = split_parent(tokenizer, cleaned, 750, 0, 900)
            for parent_index, span in enumerate(parent_spans, 1):
                text = str(span["text"]).strip()
                parent_id = f"AI_{document['slug']}_{parent_index:04d}_{sha256_text(text)[:8]}"
                parent = {
                    "id": parent_id,
                    "parent_id": parent_id,
                    "text": text,
                    **base,
                    "chapter_path": f"{source['collection']} / {document['title']}",
                    "chapter_titles": [source["collection"], document["title"]],
                    "chunk_type": "parent",
                    "parent_child_schema": "parent-child-v1",
                    "source_id": source_id,
                    "source_char_start": int(span["local_char_start"]),
                    "source_char_end": int(span["local_char_end"]),
                    "text_sha256": sha256_text(text),
                    "token_count": int(span["token_count"]),
                }
                ai_parents.append(parent)
                if args.child_strategy == "structured":
                    child_spans = split_parent_structured(
                        tokenizer,
                        text,
                        args.child_target_tokens,
                        args.child_min_tokens,
                        args.child_max_tokens,
                        args.child_overlap_max_tokens,
                        args.child_overlap_ratio_cap,
                    )
                else:
                    child_spans = split_parent(
                        tokenizer,
                        text,
                        args.child_target_tokens,
                        args.child_overlap_max_tokens,
                        args.child_max_tokens,
                    )
                for child_index, child_span in enumerate(child_spans, 1):
                    child_text = str(child_span["text"]).strip()
                    child_id = f"{parent_id}__child_{child_index:03d}"
                    child_record = {
                            "id": child_id,
                            "child_id": child_id,
                            "parent_id": parent_id,
                            "position": child_index,
                            "chunk_type": "child",
                            "parent_child_schema": "parent-child-v1",
                            "text": child_text,
                            "text_sha256": sha256_text(child_text),
                            "token_count": int(child_span["token_count"]),
                            **base,
                            "chapter_path": parent["chapter_path"],
                            "chapter_titles": parent["chapter_titles"],
                            "source_id": source_id,
                            "source_char_start": int(span["local_char_start"]) + int(child_span["local_char_start"]),
                            "source_char_end": int(span["local_char_start"]) + int(child_span["local_char_end"]),
                            "local_char_start": int(child_span["local_char_start"]),
                            "local_char_end": int(child_span["local_char_end"]),
                        }
                    child_record["retrieval_text"] = (
                        child_record["retrieval_context"] + "\n" + child_text
                    )
                    ai_children.append(child_record)
            registry.append(
                {
                    **base,
                    **license_record,
                    "path": document["path"],
                    "raw_file": str(raw_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    "cleaned_sha256": sha256_text(cleaned),
                    "cleaned_chars": len(cleaned),
                    "parent_count": len([p for p in ai_parents if p["source_slug"] == document["slug"]]),
                    "child_count": len([c for c in ai_children if c["source_slug"] == document["slug"]]),
                }
            )

    parents_dir.mkdir(parents=True, exist_ok=True)
    children_dir.mkdir(parents=True, exist_ok=True)
    (parents_dir / "ai_chunks.json").write_text(json.dumps(ai_parents, ensure_ascii=False, indent=2), encoding="utf-8")
    (children_dir / "ai_chunks.json").write_text(json.dumps(ai_children, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_dir = PROJECT_ROOT / "data/stage4/evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "topics_ai.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "query"])
        writer.writeheader()
        writer.writerows({"qid": q["qid"], "query": q["query"]} for q in eval_manifest["questions"])
    qrels = choose_qrels(eval_manifest["questions"], ai_parents)
    with (eval_dir / "qrels_ai.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "doc_id", "rel", "judgment_basis", "matched_terms", "source_slug"])
        writer.writeheader()
        writer.writerows(qrels)

    build_info = {
        "schema_version": "stage4-corpus-v1",
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "parent_target_tokens": 750,
        "child_strategy": args.child_strategy,
        "child_target_tokens": args.child_target_tokens,
        "child_min_tokens": args.child_min_tokens if args.child_strategy == "structured" else None,
        "child_max_tokens": args.child_max_tokens,
        "child_overlap_tokens": args.child_overlap_max_tokens,
        "child_overlap_ratio_cap": args.child_overlap_ratio_cap if args.child_strategy == "structured" else None,
        "legacy_config": legacy_config,
        "legacy_source_dir": str(legacy_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "hard_max_tokens": max(
            int(legacy_config.get("hard_max_tokens", legacy_config.get("child_max_tokens", 0))),
            int(args.child_max_tokens),
        ),
        "legacy_parent_hashes": legacy_parent_hashes,
        "legacy_child_hashes": legacy_child_hashes,
        "ai_parent_count": len(ai_parents),
        "ai_child_count": len(ai_children),
        "source_count": len(registry),
        "evaluation_question_count": len(eval_manifest["questions"]),
        "evaluation_qrel_count": len(qrels),
        "sources": registry,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(build_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (raw_dir / "SOURCE_REGISTRY.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: build_info[key] for key in ("source_count", "ai_parent_count", "ai_child_count", "evaluation_question_count", "evaluation_qrel_count")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
