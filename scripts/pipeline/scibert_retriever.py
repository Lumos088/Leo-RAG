import os
import json
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from scripts.pipeline.query_router import route_query

BASE_DIR = r"H:\RAG project\stage2"
VECTOR_DIR = os.path.join(BASE_DIR, "vector_db")

INDEX_PATH = os.path.join(VECTOR_DIR, "kb_scibert.index")
META_PATH = os.path.join(VECTOR_DIR, "kb_meta.json")

MODEL_NAME = "D:\\huggingface_cache\\models--allenai--scibert_scivocab_uncased\\snapshots\\24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"


class SciBertRetriever:
    def __init__(self):
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"SciBERT 索引文件不存在: {INDEX_PATH}\n"
                f"请先运行 'python scripts/pipeline/build_scibert_index.py' 构建索引"
            )
        
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.metadatas = json.load(f)
        
        print(f"正在加载 SciBERT 模型: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"SciBERT 模型加载成功，向量维度: {self.model.config.hidden_size}, 使用设备: {self.device}")

    def encode_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def search(
        self,
        query: str,
        top_k: int = 5,
        subject: str | None = None,
        lang: str | None = None
    ):
        q_emb = self.encode_text(query).astype("float32")

        candidate_k = max(top_k * 5, 20)
        scores, indices = self.index.search(q_emb, candidate_k)

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
    retriever = SciBertRetriever()

    while True:
        q = input("\n请输入问题（exit 退出）> ").strip()
        if q.lower() == "exit":
            break
        
        route = route_query(q)
        
        results = []
        top_k = 5

        if route["subject"] is not None:
            subject_priority = [route["subject"], None]
        else:
            subject_priority = [None]

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

        print("\n====== SciBERT Top-5 检索结果 ======")
        for i, r in enumerate(results, 1):
            print(f"\nTop {i} | score={r['score']:.4f}")
            print(f"  subject    : {r['subject']}")
            print(f"  lang       : {r['lang']}")
            print(f"  chunk_file : {r['chunk_file']}")
            print(f"  text       : {r['text'][:200]}...")