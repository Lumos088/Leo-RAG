# RAG Stage 2 - 计算机课程智能问答系统

一个基于检索增强生成（RAG）的计算机专业课程问答系统，支持数据结构、操作系统、计算机网络三门课程的知识检索与智能问答。

## 📁 项目结构

```
stage2/
├── app/                    # 应用程序核心模块
│   ├── rag_cli.py         # 交互式RAG问答CLI（主入口）
│   ├── bm25_retriever.py  # BM25关键词检索器
│   └── reranker.py        # BGE交叉编码器重排器
│
├── scripts/               # 数据处理与检索脚本
│   ├── data_prep/         # 数据预处理
│   │   ├── load_documents.py    # 加载原始文档
│   │   ├── clean_documents.py   # 文档清洗
│   │   ├── chunk_documents.py   # 文档分块
│   │   ├── build_vector_db.py   # 构建向量数据库
│   │   └── eval_cleaning.py     # 清洗质量评估
│   │
│   ├── pipeline/          # 检索流水线
│   │   ├── retriever.py         # FAISS向量检索器
│   │   ├── query_router.py      # 查询路由（语言/课程识别）
│   │   ├── scibert_retriever.py # SciBERT检索器
│   │   └── stats_kb.py          # 知识库统计
│   │
│   └── retrieval/         # 检索实验与评估
│       ├── run_topics.py        # 批量运行主题查询
│       ├── eval.py              # IR评估指标计算
│       ├── build_initial_pool.py    # 构建初始候选池
│       └── build_complete_pool.py   # 构建完整候选池
│
├── config/                # 配置文件
├── data/                  # 数据目录
│   ├── cleaned/          # 清洗后的文档
│   └── chunks/           # 分块后的文档
│
├── vector_db/             # 向量数据库
│   ├── kb.index          # FAISS索引文件
│   └── kb_meta.json      # 文档元数据
│
├── runs/                  # 检索结果输出
├── pools/                 # 候选池数据
├── output/                # 最终输出
├── qrels.csv             # 相关性标注（评测用）
├── topics.csv            # 查询主题（66个）
└── requirements.txt      # Python依赖
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

**依赖列表：**
- `faiss-cpu` - 向量检索库
- `sentence-transformers` - 句子嵌入模型
- `transformers` - HuggingFace模型库
- `rank_bm25` - BM25检索算法
- `jieba` - 中文分词
- `openai` - DeepSeek API客户端
- `torch` - PyTorch深度学习框架
- `numpy`, `pypdf`, `pymupdf` - 其他工具库

### 2. 配置API密钥

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY="your-api-key-here"

# 或永久添加到系统环境变量
```

### 3. 运行RAG问答

```bash
# 进入app目录
cd app

# 启动交互式CLI
python rag_cli.py
```

## 🔧 核心功能

### 1. 多种检索模式

在 `rag_cli.py` 中通过修改 `MODE` 变量切换模式：

| 模式 | 说明 |
|------|------|
| `dense` | 纯向量检索（MiniLM/SciBERT） |
| `dense_rerank` | 向量检索 + BGE重排 |
| `dense_bm25` | 向量 + BM25融合（RRF） |
| `llm_retrieval` | LLM智能检索（DeepSeek） |

### 2. 查询路由

系统自动识别：
- **语言**：中文/英文，优先返回同语言结果
- **课程**：数据结构 / 操作系统 / 计算机网络

### 3. 评估指标

```bash
cd scripts/retrieval
python eval.py
```

支持的评估指标：
- **MRR@10** - 平均倒数排名
- **Success@1/5/10** - 成功率
- **Recall@50** - 召回率
- **NDCG@10/50** - 归一化折损累积增益

## 📊 查询主题

`topics.csv` 包含66个计算机专业问题，分为4类：

| 类型 | 数量 | 示例 |
|------|------|------|
| A1 - 列举题 | 18 | "操作系统的主要功能有哪些" |
| A2 - 概念题 | 18 | "什么是 SYN Flood 攻击" |
| A3 - 定义题 | 16 | "什么是 TLB" |
| A4 - 比较题 | 14 | "TCP 和 UDP 的区别是什么" |

## 🔄 数据处理流程

```
原始文档 → 清洗 → 分块 → 向量化 → 构建索引
    ↓
用户查询 → 路由识别 → 检索 → 重排 → LLM生成答案
```

### 文档分块策略

- **目标长度**：900字符
- **硬上限**：1200字符
- **最短阈值**：200字符
- **重叠句数**：1句
- **智能分段**：按章节标题自动分割

## 📝 使用示例

### 交互式问答

```bash
$ python rag_cli.py

✅ RAG CLI（DeepSeek）启动成功 | 模式: llm_retrieval
输入问题，exit 退出。

>> 什么是死锁？

========== LLM 检索 ==========
检索结果...

================= Answer =================
死锁是指两个或多个进程在执行过程中，因争夺资源而造成的一种互相等待的现象...
```

### 批量评估

```bash
cd scripts/retrieval

# 运行所有主题查询
python run_topics.py

# 计算评估指标
python eval.py
```

## 🔬 实验对比

支持对比以下检索策略：

1. **Embedding模型对比**：MiniLM vs SciBERT
2. **融合策略对比**：纯向量 vs 向量+BM25(RRF)
3. **重排策略对比**：无重排 vs BGE重排
4. **LLM增强**：传统检索 vs LLM检索

## 📚 知识库来源

- 《操作系统导论》(OSTEP) - 中英文版
- 《现代操作系统》(Tanenbaum)
- 数据结构教材
- 计算机网络教材

## ⚙️ 配置参数

### 检索参数（rag_cli.py）

```python
MODE = "llm_retrieval"      # 检索模式
RECALL_N = 50              # 召回候选数量
FUSION_K = 20              # 融合检索数量
TOP_K = 5                  # 最终返回数量
MAX_CONTEXT_CHARS = 8000   # 上下文最大字符数
```

### 模型配置

```python
# 向量模型
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM配置
MODEL_NAME = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

## 🎯 项目特点

- ✅ 支持中英文双语问答
- ✅ 自动识别课程领域
- ✅ 多种检索策略可对比
- ✅ 完整的IR评估体系
- ✅ 基于LLM的智能检索与重排
- ✅ 模块化设计，易于扩展

## 📄 文件说明

| 文件 | 说明 |
|------|------|
| `qrels.csv` | 查询-文档相关性标注（3/2/1/0四级） |
| `topics.csv` | 66个测试查询主题 |
| `requirements.txt` | Python依赖包列表 |
| `vector_db/kb.index` | FAISS向量索引 |
| `vector_db/kb_meta.json` | 文档元数据 |

## 🔗 相关项目

- Stage 1: 数据预处理与清洗
- Stage 3: 前端界面（待开发）

---

**作者**: Boss  
**项目路径**: `H:\RAG project\stage2`
