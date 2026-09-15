import math
import os
from collections import defaultdict


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QRELS_PATH = os.path.join(PROJECT_DIR, "qrels.csv")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
SYSTEMS = ("dense", "dense_bm25", "rerank")
METRICS = ("MRR@10", "Success@1", "Success@5", "Success@10", "Recall@50", "NDCG@10", "NDCG@50")


def read_text_lines(path):
    """Read a text file while supporting the encodings used by this project."""
    last_error = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.readlines(), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def load_qrels(path):
    """Load judged qrels, excluding rows whose relevance value is empty."""
    qrels_binary = defaultdict(set)
    qrels_graded = defaultdict(dict)
    lines, encoding = read_text_lines(path)

    start_idx = 0
    for i, line in enumerate(lines):
        if line.lstrip("\ufeff").startswith("qid,doc_id,rel"):
            start_idx = i + 1
            break

    unjudged_rows = 0
    malformed_rows = 0
    all_query_ids = set()
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        try:
            qid, remainder = line.split(",", 1)
            doc_id, rel_text, _source = remainder.rsplit(",", 2)
            qid = qid.strip()
            all_query_ids.add(qid)
        except ValueError:
            malformed_rows += 1
            continue

        rel_text = rel_text.strip()
        if not rel_text:
            unjudged_rows += 1
            continue
        try:
            rel = int(float(rel_text))
        except ValueError:
            malformed_rows += 1
            continue

        doc_id = doc_id.strip().strip('"')
        qrels_graded[qid][doc_id] = rel
        if rel > 0:
            qrels_binary[qid].add(doc_id)

    stats = {
        "encoding": encoding,
        "total_queries": len(all_query_ids),
        "judged_queries": len(qrels_graded),
        "positive_queries": len(qrels_binary),
        "judged_rows": sum(len(items) for items in qrels_graded.values()),
        "unjudged_rows": unjudged_rows,
        "malformed_rows": malformed_rows,
        "no_positive_queries": sorted(set(qrels_graded) - set(qrels_binary)),
    }
    return qrels_binary, qrels_graded, stats


def load_run(path):
    run_by_doc = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid = parts[0]
            doc_id = " ".join(parts[2:-3])
            rank = int(parts[-3])
            score = float(parts[-2])
            current = run_by_doc[qid].get(doc_id)
            if current is None or rank < current["rank"]:
                run_by_doc[qid][doc_id] = {"doc_id": doc_id, "rank": rank, "score": score}
    run = defaultdict(list)
    for qid, documents in run_by_doc.items():
        run[qid] = list(documents.values())
    return run


def ranked_results(run_results, qid, k):
    return [r for r in sorted(run_results.get(qid, []), key=lambda x: x["rank"]) if r["rank"] <= k]


def calc_mrr_at_k(run_results, qrels, query_ids, k=10):
    total = 0.0
    for qid in query_ids:
        for result in ranked_results(run_results, qid, k):
            if result["doc_id"] in qrels[qid]:
                total += 1.0 / result["rank"]
                break
    return total / len(query_ids) if query_ids else 0.0


def calc_success_at_k(run_results, qrels, query_ids, k=5):
    hits = 0
    for qid in query_ids:
        retrieved = {r["doc_id"] for r in ranked_results(run_results, qid, k)}
        if retrieved & qrels[qid]:
            hits += 1
    return hits / len(query_ids) if query_ids else 0.0


def calc_recall_at_k(run_results, qrels, query_ids, k=50):
    total = 0.0
    for qid in query_ids:
        relevant_docs = qrels[qid]
        retrieved = {r["doc_id"] for r in ranked_results(run_results, qid, k)}
        total += len(retrieved & relevant_docs) / len(relevant_docs)
    return total / len(query_ids) if query_ids else 0.0


def dcg(scores):
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(scores))


def calc_ndcg_at_k(run_results, qrels_graded, query_ids, k=10):
    total = 0.0
    for qid in query_ids:
        rel_dict = qrels_graded[qid]
        gains = [rel_dict.get(r["doc_id"], 0) for r in ranked_results(run_results, qid, k)]
        ideal_gains = sorted(rel_dict.values(), reverse=True)[:k]
        idcg = dcg(ideal_gains)
        if idcg > 0:
            total += dcg(gains) / idcg
    return total / len(query_ids) if query_ids else 0.0


def evaluate(run_results, qrels_binary, qrels_graded, query_ids):
    return {
        "MRR@10": calc_mrr_at_k(run_results, qrels_binary, query_ids, 10),
        "Success@1": calc_success_at_k(run_results, qrels_binary, query_ids, 1),
        "Success@5": calc_success_at_k(run_results, qrels_binary, query_ids, 5),
        "Success@10": calc_success_at_k(run_results, qrels_binary, query_ids, 10),
        "Recall@50": calc_recall_at_k(run_results, qrels_binary, query_ids, 50),
        "NDCG@10": calc_ndcg_at_k(run_results, qrels_graded, query_ids, 10),
        "NDCG@50": calc_ndcg_at_k(run_results, qrels_graded, query_ids, 50),
    }


def main():
    print("=" * 96)
    print("IR 评估脚本 - dense vs dense_bm25 vs rerank")
    print("=" * 96)

    qrels_binary, qrels_graded, stats = load_qrels(QRELS_PATH)
    query_ids = sorted(qrels_binary)
    print(f"\nqrels 编码: {stats['encoding']}")
    print(f"有效正相关查询: {stats['positive_queries']}/{stats['total_queries']}")
    print(f"有人工判定的查询: {stats['judged_queries']}/{stats['total_queries']}")
    print(f"已判定行: {stats['judged_rows']} | 未判定行: {stats['unjudged_rows']} | 异常行: {stats['malformed_rows']}")
    if stats["no_positive_queries"]:
        print(f"没有正相关文档的已判定查询: {', '.join(stats['no_positive_queries'])}")

    runs = {}
    print("\n加载 run 文件:")
    for system_name in SYSTEMS:
        run_path = os.path.join(RUNS_DIR, f"{system_name}.run")
        if not os.path.exists(run_path):
            print(f"  {system_name}: 缺失")
            continue
        runs[system_name] = load_run(run_path)
        covered = len(set(runs[system_name]) & set(query_ids))
        print(f"  {system_name}: {len(runs[system_name])} 个查询；评测覆盖 {covered}/{len(query_ids)}")

    print("\n" + "-" * 96)
    print(f"{'System':<15} {'Coverage':<12} {'MRR@10':<10} {'S@1':<10} {'S@5':<10} {'S@10':<10} {'R@50':<10} {'NDCG@10':<10} {'NDCG@50':<10}")
    print("-" * 96)

    results_table = {}
    for system_name, run_results in runs.items():
        metrics = evaluate(run_results, qrels_binary, qrels_graded, query_ids)
        results_table[system_name] = metrics
        coverage = f"{len(set(run_results) & set(query_ids))}/{len(query_ids)}"
        print(
            f"{system_name:<15} {coverage:<12} "
            f"{metrics['MRR@10']:<10.4f} {metrics['Success@1']:<10.4f} "
            f"{metrics['Success@5']:<10.4f} {metrics['Success@10']:<10.4f} "
            f"{metrics['Recall@50']:<10.4f} {metrics['NDCG@10']:<10.4f} "
            f"{metrics['NDCG@50']:<10.4f}"
        )
    print("-" * 96)

    if "dense" in results_table:
        print("\n相对 dense 的变化:")
        for system_name in ("dense_bm25", "rerank"):
            if system_name not in results_table:
                continue
            print(f"\n  dense vs {system_name}")
            for metric in METRICS:
                base = results_table["dense"][metric]
                current = results_table[system_name][metric]
                print(f"    {metric:<12}: {base:.4f} -> {current:.4f} ({current - base:+.4f})")

    print("\n说明: 空 rel 不参与评测；缺失的 run 查询按 0 分计；NDCG 的 IDCG 使用全部已判定文档计算。")


if __name__ == "__main__":
    main()
