import json
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
META_PATH = os.path.join(BASE_DIR, "vector_db", "kb_meta.json")

with open(META_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

total_docs = len(docs)

subjects = Counter(doc.get("subject", "Unknown") for doc in docs)
langs = Counter(doc.get("lang", "Unknown") for doc in docs)
sources = Counter(doc.get("source", "Unknown") for doc in docs)
chunk_files = Counter(doc.get("chunk_file", "Unknown") for doc in docs)

print("=" * 80)
print("知识库数据统计信息")
print("=" * 80)

print(f"\n【基本信息】")
print(f"  总文档片段数: {total_docs}")

print(f"\n【课程分布】")
for subject, count in sorted(subjects.items(), key=lambda x: -x[1]):
    pct = count / total_docs * 100
    print(f"  {subject}: {count} ({pct:.2f}%)")

print(f"\n【语言分布】")
for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
    pct = count / total_docs * 100
    print(f"  {lang}: {count} ({pct:.2f}%)")

print(f"\n【来源文件统计】")
for source, count in sorted(sources.items(), key=lambda x: -x[1]):
    pct = count / total_docs * 100
    source_short = source[:50] + "..." if len(source) > 50 else source
    print(f"  {source_short}: {count} ({pct:.2f}%)")

print(f"\n【Chunk 文件统计】")
for chunk_file, count in sorted(chunk_files.items(), key=lambda x: -x[1]):
    pct = count / total_docs * 100
    print(f"  {chunk_file}: {count} ({pct:.2f}%)")

text_lengths = [len(doc.get("text", "")) for doc in docs]
avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
min_length = min(text_lengths) if text_lengths else 0
max_length = max(text_lengths) if text_lengths else 0

print(f"\n【文本长度统计】")
print(f"  平均长度: {avg_length:.2f} 字符")
print(f"  最小长度: {min_length} 字符")
print(f"  最大长度: {max_length} 字符")
print(f"  总字符数: {sum(text_lengths):,} 字符")

print("\n" + "=" * 80)
print("Markdown 表格格式")
print("=" * 80)

print("\n### 课程分布")
print("| 课程 | 文档数 | 占比 |")
print("|------|--------|------|")
for subject, count in sorted(subjects.items(), key=lambda x: -x[1]):
    pct = count / total_docs * 100
    print(f"| {subject} | {count} | {pct:.2f}% |")

print("\n### 语言分布")
print("| 语言 | 文档数 | 占比 |")
print("|------|--------|------|")
for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
    pct = count / total_docs * 100
    print(f"| {lang} | {count} | {pct:.2f}% |")

print("\n### 来源教材")
print("| 教材名称 | 文档数 | 占比 |")
print("|----------|--------|------|")
for source, count in sorted(sources.items(), key=lambda x: -x[1]):
    pct = count / total_docs * 100
    print(f"| {source} | {count} | {pct:.2f}% |")

print("\n### 文本长度统计")
print("| 指标 | 数值 |")
print("|------|------|")
print(f"| 平均长度 | {avg_length:.2f} 字符 |")
print(f"| 最小长度 | {min_length} 字符 |")
print(f"| 最大长度 | {max_length} 字符 |")
print(f"| 总字符数 | {sum(text_lengths):,} 字符 |")
