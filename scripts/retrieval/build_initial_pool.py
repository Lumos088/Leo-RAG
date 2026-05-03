import pandas as pd
from collections import defaultdict

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

def build_initial_pool():
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dense_file = os.path.join(script_dir, 'runs', 'dense.run')
    dense_bm25_file = os.path.join(script_dir, 'runs', 'dense_bm25.run')
    
    dense_runs = load_run_file(dense_file, top_n=30)
    dense_bm25_runs = load_run_file(dense_bm25_file, top_n=30)
    
    all_queries = set(dense_runs.keys()) | set(dense_bm25_runs.keys())
    
    initial_pool = defaultdict(set)
    
    for qid in all_queries:
        if qid in dense_runs:
            initial_pool[qid].update(dense_runs[qid])
        if qid in dense_bm25_runs:
            initial_pool[qid].update(dense_bm25_runs[qid])
    
    pool_data = []
    for qid in sorted(initial_pool.keys()):
        for doc_id in initial_pool[qid]:
            pool_data.append({'qid': qid, 'doc_id': doc_id})
    
    df = pd.DataFrame(pool_data)
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(script_dir, 'pools', 'initial_pool_dense_bm25.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Initial pool saved to {output_file}")
    print(f"Total queries: {len(initial_pool)}")
    print(f"Total chunks: {len(pool_data)}")
    
    for qid in sorted(initial_pool.keys())[:3]:
        print(f"\n{qid}: {len(initial_pool[qid])} chunks")
        for doc_id in list(initial_pool[qid])[:5]:
            print(f"  - {doc_id}")

if __name__ == '__main__':
    build_initial_pool()
