import os
import sys
import json
import pandas as pd
from collections import defaultdict

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(base_dir, 'venv', 'Lib', 'site-packages'))

from rank_bm25 import BM25Okapi
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_run_file(file_path, top_n=30):
    runs = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid = parts[0]
                rank = int(parts[-3])
                if rank <= top_n:
                    doc_id = ' '.join(parts[2:-3])
                    runs[qid].append(doc_id)
    return runs

def load_chunks(chunks_dir):
    all_chunks = []
    for f in os.listdir(chunks_dir):
        if f.endswith('.json'):
            filepath = os.path.join(chunks_dir, f)
            with open(filepath, 'r', encoding='utf-8') as file:
                chunks = json.load(file)
                all_chunks.extend(chunks)
    return all_chunks

class BM25Retriever:
    def __init__(self, docs):
        self.tokenized_docs = [list(jieba.cut(d["text"])) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.docs = docs

    def search(self, query, top_k=10):
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        results = []
        for doc, score in ranked[:top_k]:
            results.append({"id": doc["id"], "score": float(score)})
        return results

class TFIDFRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.texts = [d["text"] for d in docs]
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query, top_k=10):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:
                results.append({"id": self.docs[idx]["id"], "score": float(scores[idx])})
        return results

def build_complete_pool():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runs_dir = os.path.join(base_dir, 'runs')
    chunks_dir = os.path.join(base_dir, 'data', 'chunks')
    
    dense_file = os.path.join(runs_dir, 'dense.run')
    dense_bm25_file = os.path.join(runs_dir, 'dense_bm25.run')
    
    dense_runs = load_run_file(dense_file, top_n=30)
    dense_bm25_runs = load_run_file(dense_bm25_file, top_n=30)
    
    topics_file = os.path.join(base_dir, 'topics.csv')
    topics = pd.read_csv(topics_file, encoding='gbk')
    
    all_chunks = load_chunks(chunks_dir)
    bm25_retriever = BM25Retriever(all_chunks)
    tfidf_retriever = TFIDFRetriever(all_chunks)
    
    pool_sources = defaultdict(lambda: defaultdict(set))
    
    for qid in dense_runs.keys():
        for doc_id in dense_runs[qid]:
            pool_sources[qid][doc_id].add('dense')
    
    for qid in dense_bm25_runs.keys():
        for doc_id in dense_bm25_runs[qid]:
            pool_sources[qid][doc_id].add('dense_bm25')
    
    for _, row in topics.iterrows():
        qid = row['qid']
        query = row['query']
        
        bm25_results = bm25_retriever.search(query, top_k=20)
        for result in bm25_results:
            pool_sources[qid][result['id']].add('bm25')
        
        tfidf_results = tfidf_retriever.search(query, top_k=20)
        for result in tfidf_results:
            pool_sources[qid][result['id']].add('tfidf')
    
    pool_data = []
    for qid in sorted(pool_sources.keys()):
        for doc_id in sorted(pool_sources[qid].keys()):
            sources = '/'.join(sorted(pool_sources[qid][doc_id]))
            pool_data.append({
                'qid': qid,
                'doc_id': doc_id,
                'source': sources
            })
    
    df = pd.DataFrame(pool_data)
    output_file = os.path.join(base_dir, 'pools', 'complete_pool_with_sources.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Complete pool saved to {output_file}")
    print(f"Total queries: {len(pool_sources)}")
    print(f"Total chunks: {len(pool_data)}")
    
    source_stats = defaultdict(int)
    for item in pool_data:
        sources = item['source']
        source_stats[sources] += 1
    
    print("\nSource distribution:")
    for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {source}: {count}")
    
    for qid in sorted(pool_sources.keys())[:3]:
        print(f"\n{qid}: {len(pool_sources[qid])} chunks")
        for doc_id in list(pool_sources[qid].keys())[:3]:
            sources = '/'.join(sorted(pool_sources[qid][doc_id]))
            print(f"  - {doc_id} ({sources})")

if __name__ == '__main__':
    build_complete_pool()
