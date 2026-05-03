# retriever.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.pipeline.query_router import route_query

# ========= 路径 =========
BASE_DIR = r"H:\RAG project\stage2"
VECTOR_DIR = os.path.join(BASE_DIR, "vector_db")

INDEX_PATH = os.path.join(VECTOR_DIR, "kb.index")
META_PATH = os.path.join(VECTOR_DIR, "kb_meta.json")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FaissRetriever:
    def __init__(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.metadatas = json.load(f)
        self.model = SentenceTransformer(MODEL_NAME)

    def search(
        self,
        query: str,
        top_k: int = 5,
        subject: str | None = None,   # 例如 "Operating System"
        lang: str | None = None        # 例如 "zh" / "en"
    ):
        # 1) query embedding（与建库一致：normalize）
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        # 2) 先取更大的候选集
        candidate_k = max(top_k * 5, 20)
        scores, indices = self.index.search(q_emb, candidate_k)

        # 3) Python 层过滤
        results = []
        for score, idx in zip(scores[0], indices[0]):
            meta = self.metadatas[idx]

            if subject and meta.get("subject") != subject:
                continue
            if lang and meta.get("lang") != lang:
                continue

            results.append({
                "score": float(score),
                "id": meta.get("id"),
                "subject": meta.get("subject"),
                "lang": meta.get("lang"),
                "chunk_file": meta.get("chunk_file"),
                "text": meta.get("text"),
            })

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    retriever = FaissRetriever()

    while True:
        q = input("\n请输入问题（exit 退出）> ").strip()
        if q.lower() == "exit":
            break
        
        route = route_query(q)
        
        results = []
        top_k = 5

        # ===== subject 优先级 =====
        if route["subject"] is not None:
            subject_priority = [route["subject"], None]
        else:
            subject_priority = [None]

        # ===== 两层回退：subject → lang =====
        for subject in subject_priority:
            for lang in route["lang_priority"]:
                partial = retriever.search(
                    q,
                    top_k=top_k - len(results),
                    lang=lang,
                    subject=subject
                )

                seen_ids = {r["id"] for r in results}
                for r in partial:
                    if r["id"] not in seen_ids:
                        results.append(r)

                if len(results) >= top_k:
                    break
            if len(results) >= top_k:
                break

        print("\n====== Top-5 检索结果 ======")
        for i, r in enumerate(results, 1):
            print(f"\nTop {i} | score={r['score']:.4f}")
            print(f"  subject    : {r['subject']}")
            print(f"  lang       : {r['lang']}")
            print(f"  chunk_file : {r['chunk_file']}")
            print(f"  text       : {r['text'][:200]}...")
