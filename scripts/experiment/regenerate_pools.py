#!/usr/bin/env python3
"""
Regenerate pool files for chunk sizes 500-1000
使用 FAISS + SentenceTransformer (dense 模式) 对每个 query 检索 top5
每个 chunk size 使用对应的 FAISS 索引 (vector_db/experiment/corpus_XXX.index)
"""

import os
import sys
import json
import csv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ========== 路径配置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

POOL_DIR = os.path.join(BASE_DIR, "data", "pools")
EXPERIMENT_VECTOR_DIR = os.path.join(BASE_DIR, "vector_db", "experiment")
TOPICS_PATH = os.path.join(BASE_DIR, "topics.csv")

# 模型配置
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Chunk sizes to regenerate
CHUNK_SIZES = [300, 400, 500, 600, 700, 800, 900, 1000]
TOP_K = 5


def load_topics(path):
    """加载 topics.csv，尝试多种编码"""
    for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                print(f"  成功以 {encoding} 编码加载 topics.csv")
                return rows
        except (UnicodeDecodeError, csv.Error):
            continue
    raise RuntimeError(f"无法读取 topics.csv: {path}")


def detect_subject(question: str) -> str:
    """根据关键词判断课程主题"""
    q = question.lower()

    os_keywords = [
        "process", "thread", "cpu", "scheduling",
        "context switch", "memory management",
        "paging", "virtual memory",
        "deadlock", "semaphore", "mutex",
        "kernel", "interrupt", "system call",
        "死锁", "进程", "线程", "调度", "内存", "页面", "分页",
        "虚拟内存", "信号量", "互斥", "内核", "中断", "系统调用",
        "操作系统", "置换", "缓冲", "TLB", "tlb",
        "fork", "管道", "阻塞", "就绪", "作业",
    ]

    ds_keywords = [
        "stack", "queue", "heap", "tree", "binary tree", "bst",
        "graph", "hash", "hash table", "linked list", "array",
        "sorting", "search", "algorithm",
        "栈", "队列", "堆", "树", "二叉树", "图", "哈希",
        "链表", "数组", "排序", "查找", "算法",
        "数据结构", "DFS", "BFS", "遍历", "AVL",
        "红黑树", "B树", "B+树", "赫夫曼", "霍夫曼",
    ]

    cn_keywords = [
        "network", "computer network", "tcp", "udp", "ip",
        "http", "https", "dns", "osi", "osi model", "tcp/ip",
        "routing", "router", "switch", "congestion", "flow control",
        "packet", "latency", "bandwidth", "ethernet", "wifi",
        "网络", "协议", "路由", "交换", "拥塞", "流量控制",
        "数据包", "以太网", "计算机网络", "传输", "IP地址",
        "子网", "ARP", "ICMP", "SYN", "三次握手", "四次挥手",
        "Socket", "socket", "HTTP", "DNS", "FTP",
    ]

    for kw in os_keywords:
        if kw in q:
            return "Operating System"
    for kw in ds_keywords:
        if kw in q:
            return "Data Structures"
    for kw in cn_keywords:
        if kw in q:
            return "Computer Networks"
    return "Unknown"


def main():
    print("=" * 60)
    print("重新生成 pool 文件 (dense 模式, FAISS + MiniLM)")
    print("=" * 60)

    # 1) 加载 topics
    print("\n[1/4] 加载 topics.csv ...")
    topics = load_topics(TOPICS_PATH)
    print(f"  共 {len(topics)} 个查询")

    # 2) 加载模型（只加载一次）
    print("\n[2/4] 加载 SentenceTransformer 模型 ...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  模型加载成功: {MODEL_NAME}")

    # 3) 批量编码所有 queries（只编码一次，所有 chunk size 共用）
    print("\n[3/4] 批量编码所有 queries ...")
    queries_text = [t["query"] for t in topics]
    q_embeddings = model.encode(
        queries_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    print(f"  编码完成: shape={q_embeddings.shape}")

    # 4) 删除旧 pool 文件 + 对每个 chunk size 构建新 pool
    print(f"\n[4/4] 构建新 pool 文件 (top_k={TOP_K}) ...")
    os.makedirs(POOL_DIR, exist_ok=True)

    for chunk_size in CHUNK_SIZES:
        index_name = f"corpus_{chunk_size}"
        index_path = os.path.join(EXPERIMENT_VECTOR_DIR, f"{index_name}.index")
        metadata_path = os.path.join(EXPERIMENT_VECTOR_DIR, f"{index_name}_meta.json")
        pool_path = os.path.join(POOL_DIR, f"pool_{chunk_size}.json")

        print(f"\n--- chunk_size={chunk_size} ---")

        # 删除旧文件
        if os.path.exists(pool_path):
            os.remove(pool_path)
            print(f"  已删除旧文件: {pool_path}")

        # 检查索引是否存在
        if not os.path.exists(index_path):
            print(f"  跳过: 找不到索引文件 {index_path}")
            continue
        if not os.path.exists(metadata_path):
            print(f"  跳过: 找不到元数据文件 {metadata_path}")
            continue

        # 加载索引和元数据
        print(f"  加载 FAISS 索引 ...")
        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadatas = json.load(f)
        print(f"  索引向量数: {index.ntotal}, 元数据条数: {len(metadatas)}")

        # 批量检索所有 queries
        print(f"  执行批量检索 ...")
        scores, indices = index.search(q_embeddings, TOP_K)

        # 构建 pool
        pool_results = []
        for i, topic in enumerate(topics):
            results = []
            for score, idx in zip(scores[i], indices[i]):
                if idx >= 0 and idx < len(metadatas):
                    meta = metadatas[idx]
                    results.append({
                        "id": meta["id"],
                        "text": meta["text"],
                        "source": meta["source"],
                        "subject": meta["subject"],
                        "lang": meta["lang"],
                        "score": float(score),
                    })

            pool_entry = {
                "qid": topic["qid"],
                "query": topic["query"],
                "type": topic["type"],
                "subject": detect_subject(topic["query"]),
                "results": results,
            }
            pool_results.append(pool_entry)

        # 保存
        with open(pool_path, "w", encoding="utf-8") as f:
            json.dump(pool_results, f, ensure_ascii=False, indent=2)

        # 统计
        total_results = sum(len(entry["results"]) for entry in pool_results)
        all_scores = [r["score"] for entry in pool_results for r in entry["results"]]
        avg_score = np.mean(all_scores) if all_scores else 0
        print(f"  保存到: {pool_path}")
        print(f"  查询数: {len(pool_results)}, 总结果数: {total_results}, 平均score: {avg_score:.4f}")

    print("\n" + "=" * 60)
    print("所有 pool 文件重新生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
