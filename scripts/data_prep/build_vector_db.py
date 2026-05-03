# build_vector_db.py
import os
import json
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ================== 路径配置 ==================
BASE_DIR = r"H:\RAG project\stage2"

CHUNKS_DIR = os.path.join(BASE_DIR, "data", "chunks")
OUT_DIR = os.path.join(BASE_DIR, "vector_db")
os.makedirs(OUT_DIR, exist_ok=True)

# 需要纳入 KB 的 chunk 文件（你现在这三个）
CHUNK_FILES = [
    "ds_chunks.json",
    "os_chunks.json",
    "cn_chunks.json",
]

INDEX_PATH = os.path.join(OUT_DIR, "kb.index")
META_PATH = os.path.join(OUT_DIR, "kb_meta.json")
CONFIG_PATH = os.path.join(OUT_DIR, "kb_config.json")

# ================== 模型与构建参数 ==================
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64
USE_COSINE = True   # cosine 相似度（推荐）


def load_all_chunks():
    """读取多个 chunk 文件并合并"""
    all_chunks = []

    for fname in CHUNK_FILES:
        path = os.path.join(CHUNKS_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 chunk 文件：{path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"📄 加载 {fname}，chunks 数量：{len(data)}")

        # 给每个 chunk 打上来源文件，方便调试（可选但推荐）
        for item in data:
            item["chunk_file"] = fname

        all_chunks.extend(data)

    return all_chunks


def main():
    # ========= 1. 读取 chunks =========
    metadatas = load_all_chunks()

    if not metadatas:
        raise ValueError("未加载到任何 chunk，无法建库。")

    texts = [item.get("text", "") for item in metadatas]

    # 基本 sanity check
    empty_cnt = sum(1 for t in texts if not t.strip())
    if empty_cnt > 0:
        print(f"⚠️ 注意：存在 {empty_cnt} 个空 text chunk，建议回头检查 chunking。")

    # ========= 2. vector_id =========
    for i, item in enumerate(metadatas):
        item["vector_id"] = i

    # ========= 3. Embedding =========
    print("⏳ 加载 embedding 模型中...")
    model = SentenceTransformer(MODEL_NAME)

    print("⏳ 计算 embeddings（分批）...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=USE_COSINE,
    ).astype("float32")

    dim = embeddings.shape[1]

    # ========= 4. FAISS index =========
    if USE_COSINE:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    # ========= 5. 保存 =========
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    config = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "use_cosine": USE_COSINE,
        "index_type": "IndexFlatIP" if USE_COSINE else "IndexFlatL2",
        "dim": dim,
        "ntotal": int(index.ntotal),
        "chunk_files": CHUNK_FILES,
        "index_path": INDEX_PATH,
        "meta_path": META_PATH,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n✅ 向量库构建完成（多 chunks 合并）")
    print(f"总 chunks 数：{index.ntotal}")
    print("index:", INDEX_PATH)
    print("meta :", META_PATH)
    print("conf :", CONFIG_PATH)


if __name__ == "__main__":
    main()
