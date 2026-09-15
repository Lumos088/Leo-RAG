"""Configurable FAISS retriever used by the CLI, experiments and Web API."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.embedding_config import spec_from_config
from scripts.pipeline.query_router import route_query


DEFAULT_VECTOR_DIR = PROJECT_ROOT / "vector_db"


def read_faiss_index(path: Path) -> faiss.Index:
    """Read through Python to support non-ASCII Windows project paths."""
    payload = np.frombuffer(path.read_bytes(), dtype="uint8")
    return faiss.deserialize_index(payload)


class FaissRetriever:
    def __init__(
        self,
        index_dir: str | Path = DEFAULT_VECTOR_DIR,
        *,
        index_path: str | Path | None = None,
        meta_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: Optional[str] = None,
    ) -> None:
        index_dir = Path(index_dir).resolve()
        self.index_path = Path(index_path).resolve() if index_path else index_dir / "kb.index"
        self.meta_path = Path(meta_path).resolve() if meta_path else index_dir / "kb_meta.json"
        self.config_path = Path(config_path).resolve() if config_path else index_dir / "kb_config.json"

        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)
        if not self.meta_path.exists():
            raise FileNotFoundError(self.meta_path)

        config: dict[str, Any] = {}
        if self.config_path.exists():
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.embedding_spec = spec_from_config(config)

        self.index = read_faiss_index(self.index_path)
        self.metadatas = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.metadatas):
            raise ValueError(
                f"Index/meta mismatch: index={self.index.ntotal}, metadata={len(self.metadatas)}"
            )
        configured_dim = config.get("dim")
        if configured_dim is not None and int(configured_dim) != self.index.d:
            raise ValueError(f"Index dimension mismatch: config={configured_dim}, index={self.index.d}")

        self.model = SentenceTransformer(self.embedding_spec.model_name, device=device)

    def search(
        self,
        query: str,
        top_k: int = 5,
        subject: str | None = None,
        lang: str | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        query_text = self.embedding_spec.query_prefix + query.strip()
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=self.embedding_spec.normalize_embeddings,
        ).astype("float32")

        multiplier = 20 if subject or lang else 5
        candidate_k = min(max(top_k * multiplier, 20), int(self.index.ntotal))
        scores, indices = self.index.search(query_embedding, candidate_k)
        results: list[dict[str, Any]] = []
        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue
            metadata = self.metadatas[int(index_position)]
            if subject and metadata.get("subject") != subject:
                continue
            if lang and metadata.get("lang") != lang:
                continue
            result = dict(metadata)
            result["score"] = float(score)
            results.append(result)
            if len(results) >= top_k:
                break
        return results


def interactive_main() -> None:
    retriever = FaissRetriever()
    while True:
        query = input("\n请输入问题（exit 退出）> ").strip()
        if query.lower() == "exit":
            break
        route = route_query(query)
        results: list[dict[str, Any]] = []
        subject_priority = [route["subject"], None] if route["subject"] else [None]
        for subject in subject_priority:
            for language in route["lang_priority"]:
                partial = retriever.search(
                    query,
                    top_k=5 - len(results),
                    lang=language,
                    subject=subject,
                )
                seen_ids = {result["id"] for result in results}
                results.extend(result for result in partial if result["id"] not in seen_ids)
                if len(results) >= 5:
                    break
            if len(results) >= 5:
                break
        for rank, result in enumerate(results, 1):
            print(f"\nTop {rank} | score={result['score']:.4f}")
            print(f"  subject : {result.get('subject')}")
            print(f"  chapter : {result.get('chapter_path')}")
            print(f"  text    : {str(result.get('text', ''))[:200]}...")


if __name__ == "__main__":
    interactive_main()
