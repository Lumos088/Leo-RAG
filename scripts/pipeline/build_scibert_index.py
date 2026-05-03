import json
import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

BASE_DIR = r"H:\RAG project\stage2"
VECTOR_DIR = os.path.join(BASE_DIR, "vector_db")

META_PATH = os.path.join(VECTOR_DIR, "kb_meta.json")
INDEX_PATH = os.path.join(VECTOR_DIR, "kb_scibert.index")

MODEL_NAME = "D:\\huggingface_cache\\models--allenai--scibert_scivocab_uncased\\snapshots\\24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"

print("=" * 80)
print("使用 SciBERT 重建知识库索引")
print("=" * 80)

print(f"\n正在加载元数据: {META_PATH}")
with open(META_PATH, "r", encoding="utf-8") as f:
    metadatas = json.load(f)

print(f"总 chunk 数量: {len(metadatas)}")

print(f"\n正在加载 SciBERT 模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
print(f"模型加载成功，向量维度: {model.config.hidden_size}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"使用设备: {device}")

def encode_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

print("\n正在生成向量嵌入...")
embeddings = []
texts = [m["text"] for m in metadatas]

batch_size = 16
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = encode_text(batch_texts)
    embeddings.append(batch_embeddings)
    print(f"已处理 {min(i+batch_size, len(texts))}/{len(texts)} 个 chunks")

embeddings = np.vstack(embeddings).astype("float32")
print(f"\n向量矩阵形状: {embeddings.shape}")

print("\n正在构建 FAISS 索引...")
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)
print(f"索引构建完成，共 {index.ntotal} 个向量")

print(f"\n正在保存索引到: {INDEX_PATH}")
faiss.write_index(index, INDEX_PATH)
print("索引保存成功！")

print("\n验证索引...")
test_index = faiss.read_index(INDEX_PATH)
print(f"索引验证成功，共 {test_index.ntotal} 个向量")

print("\n测试检索...")
test_query = "TCP三次握手"
test_embedding = encode_text([test_query]).astype("float32")

scores, indices = test_index.search(test_embedding, 3)
print(f"\n查询: {test_query}")
for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
    meta = metadatas[idx]
    print(f"  Top {i+1}: score={score:.4f}, subject={meta['subject']}, text={meta['text'][:100]}...")

print("\n" + "=" * 80)
print("SciBERT 索引构建完成！")
print("=" * 80)