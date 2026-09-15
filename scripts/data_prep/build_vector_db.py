"""Build an isolated FAISS index for a configured embedding model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.embedding_config import MODEL_SPECS, compose_passage_text, get_embedding_spec
from scripts.pipeline.corpus_files import discover_chunk_files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def write_faiss_index(index: faiss.Index, path: Path) -> None:
    """Write through Python to support non-ASCII Windows project paths."""
    serialized = faiss.serialize_index(index)
    path.write_bytes(serialized.tobytes())


def load_all_chunks(chunks_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    chunks: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    seen_ids: set[str] = set()
    for filename in discover_chunk_files(chunks_dir):
        path = chunks_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list: {path}")
        for item in data:
            chunk_id = str(item.get("id") or "")
            if not chunk_id:
                raise ValueError(f"Chunk without id in {path}")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk id: {chunk_id}")
            seen_ids.add(chunk_id)
            record = dict(item)
            record["chunk_file"] = filename
            chunks.append(record)
        hashes[filename] = sha256(path)
        print(f"loaded {filename}: {len(data)} chunks")
    return chunks, hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="minilm")
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "stage1" / "chunks",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--text-mode", choices=("raw", "contextual"), default="contextual")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_embedding_spec(args.model)
    chunks_dir = args.chunks_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else PROJECT_ROOT / "vector_db" / "stage1" / spec.key / args.text_mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "kb.index"
    meta_path = output_dir / "kb_meta.json"
    config_path = output_dir / "kb_config.json"
    existing = [path for path in (index_path, meta_path, config_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists under {output_dir}; pass --overwrite to rebuild"
        )

    metadatas, chunk_hashes = load_all_chunks(chunks_dir)
    texts = [compose_passage_text(item, args.text_mode) for item in metadatas]
    empty_ids = [item["id"] for item, text in zip(metadatas, texts) if not text]
    if empty_ids:
        raise ValueError(f"Empty embedding text for {len(empty_ids)} chunks; first={empty_ids[0]}")

    encoded_texts = [spec.passage_prefix + text for text in texts]
    print(f"loading model: {spec.model_name}")
    load_started = time.perf_counter()
    model = SentenceTransformer(spec.model_name, device=args.device)
    model_load_seconds = time.perf_counter() - load_started

    encode_started = time.perf_counter()
    embeddings = model.encode(
        encoded_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=spec.normalize_embeddings,
    ).astype("float32")
    encode_seconds = time.perf_counter() - encode_started

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension) if spec.normalize_embeddings else faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    for vector_id, item in enumerate(metadatas):
        item["vector_id"] = vector_id

    write_faiss_index(index, index_path)
    meta_path.write_text(json.dumps(metadatas, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_embeddings:
        np.save(output_dir / "embeddings.npy", embeddings)

    spec_config = spec.to_dict()
    spec_config["model_key"] = spec_config.pop("key")
    config = {
        "schema_version": "stage1-index-v1",
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **spec_config,
        "text_mode": args.text_mode,
        "batch_size": args.batch_size,
        "device": args.device,
        "index_type": "IndexFlatIP" if spec.normalize_embeddings else "IndexFlatL2",
        "dim": dimension,
        "ntotal": int(index.ntotal),
        "chunks_dir": display_path(chunks_dir),
        "chunk_files": list(discover_chunk_files(chunks_dir)),
        "chunk_file_sha256": chunk_hashes,
        "index_path": display_path(index_path),
        "meta_path": display_path(meta_path),
        "model_load_seconds": round(model_load_seconds, 3),
        "encode_seconds": round(encode_seconds, 3),
        "chunks_per_second": round(len(metadatas) / encode_seconds, 3),
        "index_bytes": index_path.stat().st_size,
        "metadata_bytes": meta_path.stat().st_size,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"built: model={spec.key} chunks={index.ntotal} dim={dimension}")
    print(f"index: {index_path}")
    print(f"config: {config_path}")


if __name__ == "__main__":
    main()
