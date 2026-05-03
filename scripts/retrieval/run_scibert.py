import sys
import os
import csv
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from scripts.pipeline.scibert_retriever import SciBertRetriever
from app.bm25_retriever import BM25Retriever
from app.reranker import BGECrossEncoderReranker

TOPICS_PATH = os.path.join(parent_dir, "topics.csv")
META_PATH = os.path.join(parent_dir, "vector_db", "kb_meta.json")
RUN_PATH = os.path.join(parent_dir, "runs")

FUSION_K = 50
TOP_K = 50
RECALL_N = 50

def load_doc_contents():
    doc_contents = {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
        for doc in docs:
            doc_id = doc["id"]
            doc_contents[doc_id] = doc.get("text", "")
    return doc_contents

def rrf(rank, k=60):
    return 1.0 / (k + rank)

def load_topics():
    topics = []
    with open(TOPICS_PATH, "r", encoding="gbk") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topics.append({
                "qid": row["qid"],
                "query": row["query"],
                "type": row["type"]
            })
    return topics

def run_scibert(retriever, query, top_k=TOP_K):
    results = retriever.search(query, top_k=top_k)
    return results

def run_scibert_bm25(retriever, bm25_retriever, query, top_k=TOP_K, fusion_k=FUSION_K):
    scibert_results = retriever.search(query, top_k=fusion_k)
    bm25_results = bm25_retriever.search(query, top_k=fusion_k)

    merged = {}

    for rank, r in enumerate(scibert_results, 1):
        rid = r["id"]
        merged.setdefault(rid, dict(r))
        merged[rid]["scibert_score"] = r.get("score")
        merged[rid]["scibert_rank"] = rank
        merged[rid]["rrf_score"] = merged[rid].get("rrf_score", 0.0) + rrf(rank)

    for rank, r in enumerate(bm25_results, 1):
        rid = r["id"]
        merged.setdefault(rid, dict(r))
        merged[rid]["bm25_score"] = r.get("bm25_score")
        merged[rid]["bm25_rank"] = rank
        merged[rid]["rrf_score"] = merged[rid].get("rrf_score", 0.0) + rrf(rank)

    results = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]
    return results

def run_scibert_rerank(retriever, bm25_retriever, reranker, doc_contents, query, top_k=TOP_K, fusion_k=FUSION_K):
    scibert_results = retriever.search(query, top_k=fusion_k)
    bm25_results = bm25_retriever.search(query, top_k=fusion_k)

    merged = {}

    for rank, r in enumerate(scibert_results, 1):
        rid = r["id"]
        merged.setdefault(rid, dict(r))
        merged[rid]["scibert_score"] = r.get("score")
        merged[rid]["scibert_rank"] = rank
        merged[rid]["rrf_score"] = merged[rid].get("rrf_score", 0.0) + rrf(rank)

    for rank, r in enumerate(bm25_results, 1):
        rid = r["id"]
        merged.setdefault(rid, dict(r))
        merged[rid]["bm25_score"] = r.get("bm25_score")
        merged[rid]["bm25_rank"] = rank
        merged[rid]["rrf_score"] = merged[rid].get("rrf_score", 0.0) + rrf(rank)

    candidates = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)[:fusion_k]
    
    for c in candidates:
        c["text"] = doc_contents.get(c["id"], "")
    
    reranked = reranker.rerank(query, candidates, top_k=top_k)
    return reranked

def save_run(results, qid, system_name, run_path):
    os.makedirs(run_path, exist_ok=True)
    run_file = os.path.join(run_path, f"{system_name}.run")
    
    mode = "a" if os.path.exists(run_file) else "w"
    
    with open(run_file, mode, encoding="utf-8") as f:
        for rank, r in enumerate(results, 1):
            if system_name == "scibert":
                score = r.get("score", 0.0)
            elif system_name == "scibert_bm25":
                score = r.get("rrf_score", 0.0)
            elif system_name == "scibert_rerank":
                score = r.get("rerank_score", 0.0)
            else:
                score = 0.0
            
            doc_id = r["id"]
            f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {system_name}\n")

def main():
    print("=" * 60)
    print("SciBERT 对比实验 - 生成 TREC run 文件")
    print("=" * 60)
    
    print("\n[1/4] 加载 topics...")
    topics = load_topics()
    print(f"    共加载 {len(topics)} 个问题")
    
    print("\n[2/4] 初始化检索器...")
    scibert_retriever = SciBertRetriever()
    print("    SciBERT 检索器已初始化")
    
    with open(META_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    bm25_retriever = BM25Retriever(docs)
    print("    BM25 检索器已初始化")
    
    reranker = BGECrossEncoderReranker()
    print("    Reranker 已初始化")
    
    print("\n[3/4] 加载文档内容...")
    doc_contents = load_doc_contents()
    print(f"    共加载 {len(doc_contents)} 个文档")
    
    print("\n[4/4] 批量检索并保存结果...")
    for i, topic in enumerate(topics, 1):
        qid = topic["qid"]
        query = topic["query"]
        
        print(f"    [{i}/{len(topics)}] {qid}: {query[:30]}..." if len(query) > 30 else f"    [{i}/{len(topics)}] {qid}: {query}")
        
        # SciBERT 纯检索
        scibert_results = run_scibert(scibert_retriever, query, top_k=TOP_K)
        save_run(scibert_results, qid, "scibert", RUN_PATH)
        
        # SciBERT + BM25 融合
        scibert_bm25_results = run_scibert_bm25(scibert_retriever, bm25_retriever, query, top_k=TOP_K, fusion_k=FUSION_K)
        save_run(scibert_bm25_results, qid, "scibert_bm25", RUN_PATH)
        
        # SciBERT + Rerank
        scibert_rerank_results = run_scibert_rerank(scibert_retriever, bm25_retriever, reranker, doc_contents, query, top_k=TOP_K, fusion_k=FUSION_K)
        save_run(scibert_rerank_results, qid, "scibert_rerank", RUN_PATH)
    
    print("\n" + "=" * 60)
    print("完成！")
    print(f"run 文件保存在: {RUN_PATH}")
    print("  - scibert.run")
    print("  - scibert_bm25.run")
    print("  - scibert_rerank.run")
    print("=" * 60)

if __name__ == "__main__":
    main()