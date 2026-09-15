import sys
import os
import csv
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from scripts.pipeline.retriever import FaissRetriever
from scripts.pipeline.hybrid_retrieval import hybrid_rerank, hybrid_search
from app.bm25_retriever import BM25Retriever
from app.reranker import BGECrossEncoderReranker


TOPICS_PATH = os.path.join(parent_dir, "topics.csv")
META_PATH = os.path.join(parent_dir, "vector_db", "kb_meta.json")
RUN_PATH = os.path.join(parent_dir, "runs")

TOP_K = 50

# 读取文档内容
def load_doc_contents():
    doc_contents = {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
        for doc in docs:
            doc_id = doc["id"]
            doc_contents[doc_id] = doc.get("text", "")
    return doc_contents

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

def run_dense(retriever, query, top_k=TOP_K):
    results = retriever.search(query, top_k=top_k)
    return results

def run_dense_bm25(retriever, bm25_retriever, query, top_k=TOP_K):
    return hybrid_search(retriever, bm25_retriever, query)[:top_k]

def run_rerank(retriever, bm25_retriever, reranker, query, top_k=TOP_K):
    return hybrid_rerank(retriever, bm25_retriever, reranker, query, top_k)

def save_run(results, qid, system_name, run_path):
    os.makedirs(run_path, exist_ok=True)
    run_file = os.path.join(run_path, f"{system_name}.run")
    
    mode = "a" if os.path.exists(run_file) else "w"
    
    with open(run_file, mode, encoding="utf-8") as f:
        for rank, r in enumerate(results, 1):
            if system_name == "dense":
                score = r.get("score", 0.0)
            elif system_name == "dense_bm25":
                score = r.get("rrf_score", 0.0)
            elif system_name == "rerank":
                score = r.get("rerank_score", 0.0)
            else:
                score = 0.0
            
            doc_id = r["id"]
            f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {system_name}\n")

def main():
    print("=" * 60)
    print("批量检索脚本 - 生成 TREC run 文件")
    print("=" * 60)
    
    print("\n[1/5] 加载 topics...")
    topics = load_topics()
    print(f"    共加载 {len(topics)} 个问题")
    
    print("\n[2/5] 初始化检索器...")
    retriever = FaissRetriever()
    print("    Dense 检索器已初始化")
    
    with open(META_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    bm25_retriever = BM25Retriever(docs)
    print("    BM25 检索器已初始化")
    
    reranker = BGECrossEncoderReranker()
    print("    Reranker 已初始化")
    
    print("\n[3/5] 加载文档内容...")
    doc_contents = load_doc_contents()
    print(f"    共加载 {len(doc_contents)} 个文档")
    
    print("\n[4/5] 批量检索并保存结果...")
    os.makedirs(RUN_PATH, exist_ok=True)
    for system_name in ("dense", "dense_bm25", "rerank"):
        with open(os.path.join(RUN_PATH, f"{system_name}.run"), "w", encoding="utf-8"):
            pass

    for i, topic in enumerate(topics, 1):
        qid = topic["qid"]
        query = topic["query"]
        
        print(f"    [{i}/{len(topics)}] {qid}: {query[:30]}..." if len(query) > 30 else f"    [{i}/{len(topics)}] {qid}: {query}")
        
        # Dense 检索
        dense_results = run_dense(retriever, query, top_k=TOP_K)
        save_run(dense_results, qid, "dense", RUN_PATH)
        
        # Dense + BM25 融合
        dense_bm25_results = run_dense_bm25(retriever, bm25_retriever, query, top_k=TOP_K)
        save_run(dense_bm25_results, qid, "dense_bm25", RUN_PATH)
        
        # Rerank
        rerank_results = run_rerank(retriever, bm25_retriever, reranker, query, top_k=TOP_K)
        save_run(rerank_results, qid, "rerank", RUN_PATH)
        
    
    print("\n[5/5] 完成！")
    print(f"run 文件保存在: {RUN_PATH}")
    print("  - dense.run")
    print("  - dense_bm25.run")
    print("  - rerank.run")
    print("=" * 60)

if __name__ == "__main__":
    main()
