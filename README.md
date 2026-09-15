# RAG Intelligent Q&A System

面向课程学习的证据增强智能问答系统。当前知识库覆盖操作系统、计算机网络、数据结构，以及人工智能、机器学习、深度学习和 LLM 相关资料。

系统使用结构化 Parent-Child Chunking、Dense + BM25 混合检索、RRF 融合、BGE Cross-Encoder 重排和 DeepSeek 流式生成，并提供多会话、单会话记忆、引用溯源及检索过程展示。

> 当前生产配置位于 `config/parent_child.json` 和 `config/retrieval.json`。程序路径均根据项目根目录自动定位，不依赖 `E:\RAG` 这一固定路径。

## 核心能力

- 中文、英文及中英混合课程问答
- 操作系统、计算机网络、数据结构与 AI 资料的主题路由
- Dense 与 BM25 并行召回，使用 RRF 融合
- BGE Cross-Encoder Child 重排与 Parent 级重排
- Parent-Child 检索：Child 负责精确定位，Parent 负责完整上下文
- 基线证据补位与定义题 Child 高亮
- 多会话创建、切换、重命名、清空和删除
- 单个会话内的上下文记忆和指代消解
- SQLite 持久化、WAL 并发控制和 request_id 幂等写入
- SSE 流式生成及 Pipeline 阶段状态同步
- Light / Dark 主题和响应式 Streamlit 界面
- qrels、开发集、冻结 holdout 与答案级评测

## 当前系统架构

```text
Streamlit Frontend
        │ question / conversation_id / request_id / settings
        ▼
FastAPI
        │
        ├── SQLite Conversation Store
        │      └── recent messages + rolling summary
        │
        ├── Context-dependent Query Rewrite
        ├── Subject Router
        │
        ├── Child Dense Retrieval ─┐
        ├── Child BM25 Retrieval ─┴── RRF Fusion
        │                              │
        │                         BGE Child Rerank
        │                              │
        │                 Parent Aggregation / Parent RRF
        │                              │
        │             Baseline Rescue + Child Highlight
        │                              │
        └────────────────────── Context Assembly
                                       │
                                DeepSeek Generation
                                       │ SSE
                                       ▼
                         Answer + Citations + Retrieval Results
```

### 前端

`web_ui/streamlit_ui.py` 负责：

- Conversation 与 Settings 两种侧栏视图
- 历史会话管理、示例问题和消息展示
- 引用标签及检索结果折叠区
- Query、Dense、BM25、Fusion、BGE、Context、LLM、Answer 状态展示
- Light / Dark 主题

### 后端

- `web_ui/main.py`：FastAPI 接口、请求校验、会话 CRUD 和 SSE 输出
- `web_ui/api/rag_service.py`：记忆、问题改写、检索、上下文组装和生成编排
- `web_ui/api/conversation_store.py`：SQLite 会话、消息、引用和摘要持久化
- `web_ui/api/memory.py`：最近消息、滚动摘要和上下文相关问题改写
- `scripts/pipeline/parent_child_retrieval.py`：Child 检索与 Parent 聚合/重排
- `scripts/pipeline/context_assembly.py`：证据补位、Child 高亮和上下文预算控制

## Parent-Child Chunking

单层 Chunk 需要同时满足两个互相冲突的目标：

- Chunk 小：检索定位更精确，但定义背景、条件列表或例子容易被切断。
- Chunk 大：上下文更完整，但关键词和向量表示容易被其他内容稀释。

当前系统将两种职责分开：

1. Parent 保留课程、文档、章节路径和完整正文。
2. Parent 按章节、段落和句子边界切成较短 Child。
3. Dense 与 BM25 在 Child 层检索。
4. 多个命中 Child 按 `parent_id` 聚合，避免同一章节重复占据最终名额。
5. LLM 阅读完整 Parent，同时保留代表 Child 解释命中位置。

### 当前分块参数

| 语料 | Child 目标/上限 | 最小长度 | 最大 overlap | overlap 比例上限 |
| --- | ---: | ---: | ---: | ---: |
| 操作系统、计算机网络、数据结构 | 205 tokens | 30 | 50 tokens | 0.244 |
| AI 扩充资料 | 70 tokens | 30 | 15 tokens | 0.21 |

### 参数如何确定

`205/50` 不是只凭经验指定，也不是简单比较 100、200、300、400。候选范围先由语料统计和模型容量推导，再进行对比实验。

关键观测：

- 段落长度：P75=29、P95=51 tokens
- 句子长度：P50=17、P75=27、P95=47 tokens
- MiniLM 最大输入：128 tokens，扣除特殊 token 后裸正文容量约 126
- AI 元数据前缀：P95=53 tokens，带前缀时安全正文容量约 73
- 标准证据跨度：P50=479、P95=652 tokens，因此 Child 不应直接承担完整答案

由此生成三轮共 21 组候选：

| 候选区间 | Chunk 候选 | Overlap 候选 | 依据 |
| --- | --- | --- | --- |
| 安全区 | 40 / 50 / 70 | 0 / 15 / 25 | 自然段分布与带元数据的安全容量 |
| 容量区 | 80 / 100 / 125 | 0 / 25 | MiniLM 裸正文容量 |
| 混合检索区 | 155 / 205 / 250 | 25 / 50 | Dense 软上界，以及 BM25/BGE 可读取完整 Child |

课程资料最终选择 `205/50`。虽然 `205/25` 的部分纯检索指标略高，但 `205/50` 在答案级评测中对跨边界事实更稳定、P95 延迟更低，并在开发集与冻结 holdout 上通过完整性门槛。

AI 资料采用 `70/15`，主要由 MiniLM 128-token 容量和 AI 元数据前缀 P95=53 推导，之后使用 25 道 AI 题进行回归验证。

## Embedding 模型选择

项目在相同语料、查询和 Hybrid 管线下对比：

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `intfloat/multilingual-e5-base`
- `BAAI/bge-m3`

选择规则：先保留 Hybrid Recall@50 不低于最好结果 95% 的候选，再比较排序质量与资源成本。

| 模型 | 维度 | 索引大小 | Hybrid Recall@50 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniLM | 384 | 11.7 MB | **0.9493** | 0.9297 | 0.6395 |
| multilingual-e5-base | 768 | 23.3 MB | 0.8130 | 0.9225 | 0.6638 |
| BGE-M3 | 1024 | 31.1 MB | 0.8890 | **0.9453** | **0.6959** |

MiniLM 在当前语料上具有最高的 Hybrid Recall@50 和最小索引，因此作为生产 Embedding。BGE-M3 的头部排序更好，但未通过召回门槛且索引更大。

这不代表 MiniLM 普遍优于 BGE-M3，只代表它更适合当前语料与评测集。BGE 模型仍用于 Cross-Encoder 重排，因为召回模型与重排模型承担不同任务。

## 检索参数

当前核心配置：

```json
{
  "dense_k": 75,
  "bm25_k": 50,
  "rrf_k": 20,
  "dense_weight": 1.0,
  "bm25_weight": 1.0,
  "rerank_candidates": 50,
  "parent_rerank_candidates": 20,
  "parent_rrf_k": 20,
  "parent_top_k": 6,
  "context_token_budget": 6000,
  "max_context_chars": 16000
}
```

### 45 组融合参数如何得到

- 5 组 Dense/BM25 深度：`(30,50)`、`(50,50)`、`(50,75)`、`(75,50)`、`(75,75)`
- 3 个 RRF k：`20`、`60`、`100`
- 3 个 Dense 权重：`0.5`、`1.0`、`1.5`，BM25 权重固定为 1
- Rerank candidates 固定为 50

总组合数为 `5 × 3 × 3 = 45`。

64 个有标注问题按学科稳定分成开发集 48 题与测试集 16 题。开发集先要求 Recall@50 不低于基线 0.9479，再按 NDCG@10 筛选；只有 4 组进入 BGE 复评。最终配置还必须在测试集 NDCG@10 和 Recall@50 上都不低于基线。

最终选择 `Dense K=75 / BM25 K=50 / RRF k=20 / 权重 1:1 / Rerank=50`：

- 测试集 NDCG@10：0.7907 → 0.7965（+0.0058）
- 测试集 Recall@50：0.9449 → 0.9477（+0.0027）

`rerank_candidates=50` 和 `parent_rerank_candidates=20` 是经过回归验证的工程成本边界，但没有分别进行完整网格搜索，不能表述为数学最优值。

### Top-K 如何确定

需要区分候选召回 K 与最终上下文 Top-K：

- Dense K=75、BM25 K=50：构建候选池。
- 前端/API 默认 Top-K=5：控制交互中的最终返回数量，可在 1–10 之间调整。
- Parent-Child 离线验收重点使用 K=6：复杂题的事实覆盖和完整率更稳定。
- Baseline rescue：仅在 Parent-Child 漏掉强基线证据时最多补入 1 个不同 Parent。
- 最终上下文仍受 6000 tokens 和 16000 字符上限约束。

## 会话与记忆

- 会话、消息、引用、检索元数据和摘要保存在 `data/runtime/conversations.db`。
- 最近 8 条消息直接参与记忆准备，更早内容被压缩为最多 3000 字符摘要。
- 只有存在指代词或短追问时才改写为独立检索问题。
- 问题改写和摘要温度为 0，降低随机变化。
- 历史记忆只用于解析指代，回答事实仍必须来自本轮检索材料。
- SQLite 使用 WAL、10 秒连接超时和 `busy_timeout`。
- `request_id` 唯一约束防止流式重试产生重复消息。

最近 8 条、摘要 3000 字符和生成温度 0.2 属于经过功能回归的工程默认值，尚未作为独立变量进行全面参数搜索。

## 快速开始

### 1. 安装依赖

```powershell
cd E:\RAG
python -m pip install -r .\requirements.txt
python -m pip install -r .\web_ui\requirements.txt
```

NVIDIA GPU 环境可在基础依赖之后安装项目锁定的 CUDA 版 PyTorch：

```powershell
python -m pip install -r .\requirements-cuda.txt
```

验证 CUDA：

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### 2. 配置 DeepSeek API Key

当前 PowerShell 会话临时设置：

```powershell
$env:DEEPSEEK_API_KEY="your-api-key"
```

永久写入当前 Windows 用户环境变量：

```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "your-api-key", "User")
```

设置后请重新打开 PowerShell。不要输出完整密钥，可使用以下命令安全验证：

```powershell
if ($env:DEEPSEEK_API_KEY) { "DEEPSEEK_API_KEY 已设置，长度：$($env:DEEPSEEK_API_KEY.Length)" } else { "未设置" }
```

### 3. 一键启动前后端

```powershell
cd E:\RAG
python .\start.py
```

默认地址：

- 前端：<http://localhost:8501>
- 后端健康检查：<http://127.0.0.1:8000/api/health>

不自动打开浏览器：

```powershell
python .\start.py --no-browser
```

自定义端口：

```powershell
python .\start.py --api-port 8001 --ui-port 8502
```

按 `Ctrl+C` 同时关闭前端和后端。

## 评测指标

| 指标 | 含义 | 主要用途 |
| --- | --- | --- |
| 向量相似度 | 问题与单条 Child 的语义接近程度 | Dense 局部召回信号 |
| BM25 | 关键词重合与稀有词匹配程度 | 词法召回信号 |
| Recall@K | 所有已标正例中，前 K 找到了多少 | 判断候选池是否漏证据 |
| Success@K | 每题前 K 是否至少命中一个正例 | 判断是否存在可用答案入口 |
| MRR | 第一个相关结果出现得有多靠前 | 评价首个答案入口位置 |
| NDCG@K | 同时考虑相关等级与排序位置 | 评价前部整体排序质量 |
| Precision@K | 前 K 中正例的比例 | qrels 不完整时仅视为下界 |
| Fact Coverage | 标准答案关键事实覆盖比例 | 学习问答的核心答案指标 |
| Complete Answer Rate | 问题要求的要点是否全部回答 | 检查综合题是否完整 |
| Citation Correctness | 引用是否支持对应结论 | 检查证据可追溯性 |
| Unsupported Claims | 回答中是否存在材料外断言 | 检查幻觉和越界 |
| Latency / tokens / redundancy | 延迟、成本和重复上下文 | 质量门槛后的效率约束 |

## 验证与测试

运行全部自动化测试：

```powershell
cd E:\RAG
python -m pytest -q
```

按阶段运行：

```powershell
python -m pytest .\tests\test_parent_child_runtime.py -q
python -m pytest .\tests\test_stage3_conversations.py -q
python -m pytest .\tests\test_stage4_corpus.py -q
python -m pytest .\tests\test_chunking_strategy.py -q
```

验证 Parent-Child 数据与引用关系：

```powershell
python .\scripts\experiment\verify_parent_child.py `
  --parent-child-dir .\data\stage4\parent_child\minilm_struct205_o50_ai70_o15_v1
```

运行当前 AI 资料评测：

```powershell
python .\scripts\experiment\evaluate_stage4_ai.py `
  --config .\config\parent_child.json `
  --device cuda `
  --top-k 6 `
  --output-dir .\experiments\stage4\ai_eval_production
```

主要验收结果：

- Chunk 开发集 15 题：Fact Coverage 0.8944 → 0.9028，完整率不退化
- 冻结 holdout 15 题：Fact Coverage 0.9438 → 0.9756，完整率 0.7333 → 0.8667
- AI 25 题：Recall 0.90 → 0.94，NDCG 0.7192 → 0.7688
- 检索测试集 16 题：NDCG@10 +0.0058，Recall@50 +0.0027

## 项目结构

```text
RAG/
├── start.py                         # FastAPI + Streamlit 一键启动
├── app/                             # CLI 与基础检索组件
├── web_ui/                          # FastAPI、Streamlit、会话与记忆
├── scripts/
│   ├── data_prep/                   # 清洗、分块、索引和语料构建
│   ├── pipeline/                    # 检索、路由与上下文组装
│   ├── retrieval/                   # 召回池与 IR 评测
│   └── experiment/                  # 参数搜索和分阶段验证
├── config/                          # 生产与参考配置
├── data/                            # 原始、清洗、Parent/Child 与运行数据
├── vector_db/                       # FAISS 索引与元数据
├── experiments/                     # 实验结果和冻结报告
├── tests/                           # 自动化回归测试
├── topics.csv                       # 评测问题
└── qrels.csv                        # 分级相关性标注
```

## 数据规模

| 项目 | 当前数量 |
| --- | ---: |
| Parent | 7706 |
| Child | 21544 |
| 原三门课程 Parent | 7587 |
| 新增 AI Parent | 119 |
| 新增 AI Child | 1536 |
| 来源文档 | 21 |

旧 Parent ID 保持稳定，新增资料通过来源注册、许可证记录、文件哈希和版本化索引管理。

## 参数证据等级

1. **实验选优**：在明确候选空间和评测集上选出，例如课程 `205/50`、检索 `75/50/RRF20`、Parent RRF20。
2. **统计推导并回归**：由语料分位数或模型容量确定候选，再做专项验证，例如 AI `70/15`、`min_tokens=30`。
3. **工程默认或安全约束**：经过功能回归但没有独立全面寻优，例如候选重排 50、Parent 候选 20、上下文 6000 tokens、最近 8 条消息、摘要 3000 字符和生成温度 0.2。

只有第一类可以描述为“在当前候选空间和评测集上的最佳”。语料、Embedding 模型或延迟预算变化后，所有参数都需要重新验证。

## 版本演进

- **旧版**：单层固定 Chunk、单轮问答、平铺 Streamlit 和硬编码路径。
- **Stage 1**：路径自动定位、Chunk 元数据、64 题 qrels、Embedding 对比和检索参数调优。
- **Stage 2**：Parent-Child、Parent 聚合/重排、证据补位和答案级评测。
- **Stage 3**：多会话、上下文记忆、问题改写、SQLite、SSE 和幂等写入。
- **Stage 4**：扩充 AI 资料，基于语料分布和模型容量生成 21 组 Chunk 候选，并通过开发集与冻结 holdout 验证。

未形成稳定生产收益的 SciBERT 与 LLM Retrieval 已从代码和前端移除。Embedding 微调和复杂 Agentic Retrieval 暂未加入；只有当现有检索、分块和证据组装仍表现出可量化瓶颈时再考虑。

## 相关文档

- `RAG系统架构与优化总结.docx`：当前架构、原理、参数依据、指标和上线判据
- `RAG系统版本演进与优化说明.docx`：旧版本到当前版本的优化过程
- `RAG系统改动实施说明.docx`：阶段任务与实施说明
- `STAGE2_IMPLEMENTATION_REPORT.md`：Parent-Child 实施记录
- `STAGE3_IMPLEMENTATION_REPORT.md`：多会话与记忆实施记录
- `STAGE4_IMPLEMENTATION_REPORT.md`：AI 资料扩充与 Chunk 调优记录
- `THIRD_PARTY_NOTICES_STAGE4.md`：第三方资料来源与使用条件

## 安全与隐私

- `DEEPSEEK_API_KEY` 只应通过环境变量提供，不要写入代码、README 或提交到 Git。
- 提交给 DeepSeek API 的内容可能包括用户问题、必要的会话记忆和检索到的课程证据。
- 历史会话保存在本机 SQLite 数据库中。
- 会话记忆不是事实来源；最终回答仍应由检索证据支持。

## License

项目代码及数据资料的许可范围可能不同。使用或分发知识库资料前，请检查 `THIRD_PARTY_NOTICES_STAGE4.md` 和各来源的许可证记录。
