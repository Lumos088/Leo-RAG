# Stage 4 AI 知识库与数据驱动分块实施报告

## 结论

生产环境已切换到结构边界优先的 Parent-Child 分块。旧课程保持原 Parent ID；新增 AI 文献使用更短的 Child，以避免元数据前缀挤占 MiniLM 的有效输入。最终配置为：

- 旧课程：结构化可变长度 Child，目标与上限 205 tokens，最小 30 tokens，动态 overlap 上限 50 tokens、比例上限 24.4%。
- AI 文献：结构化 Child，目标与上限 70 tokens，最小 30 tokens，动态 overlap 上限 15 tokens、比例上限 21%。
- 检索：Dense 75、BM25 50、RRF 20、BGE 候选 50、Parent Top-K 6。
- 语料：7,706 个 Parent、21,544 个 Child；其中新增 119 个 AI Parent、1,536 个 AI Child。

## 参数不是凭经验直接定值

先统计真实语料、证据跨度和模型容量，再派生候选范围。MiniLM 最大输入为 128 tokens，扣除特殊 token 后原始正文容量约 126 tokens；AI 文献的检索前缀 p95 为 53 tokens，因此 AI Child 的共享安全正文容量约 73 tokens。语料段落 token 的 p90/p95 为 41/51，句子 p50/p75 为 17/27。

实验分三轮共筛选 21 组结构化参数：

1. 安全容量区间：40、50、70 tokens，overlap 0、15、25。
2. 原始正文容量区间：80、100、125 tokens，overlap 0、25。
3. 混合检索软上限区间：155、205、250 tokens，overlap 25、50。

第三轮允许 Child 超过 Dense 的 126-token 容量，因为 BM25 和 BGE 会读取完整 Child；上限仍限制为约 2 倍 Dense 容量，防止正文过长稀释语义。最终没有选检索单项最高的 205/25，而选 205/50：后者在答案级验证中补回了边界证据，延迟更低，并在 Parent Top-K=6 时通过完整性门槛。

## 结构化分块规则

分块优先使用段落或换行边界；单段过长时再使用句子边界；只有无法满足上限时才退化为 token 窗口。overlap 是上限而不是固定复制量，实际重叠同时受相邻结构边界和比例上限约束。这样既减少标题、定义、列表项被硬切开的概率，也避免固定 50-token 重叠在短 Child 上造成过度重复。

## 验收结果

### 开发集答案级对照 15 题

| 指标 | 旧生产方案 | 新候选 | 变化 |
|---|---:|---:|---:|
| 必备事实覆盖率 | 0.8944 | 0.9028 | +0.0083 |
| 完整答案率 | 0.7333 | 0.7333 | +0.0000 |
| 引用有效率 | 1.0000 | 1.0000 | +0.0000 |
| 平均无依据主张数 | 0.0000 | 0.0000 | +0.0000 |
| 平均上下文字符数 | - | 5,857 | 低于 16,000 上限 |

偏好结果为新候选 8、平局 6、旧方案 1。

### 冻结 Holdout 15 题

| 指标 | 旧生产方案 | 新候选 | 变化 |
|---|---:|---:|---:|
| 必备事实覆盖率 | 0.9438 | 0.9756 | +0.0317 |
| 完整答案率 | 0.7333 | 0.8667 | +0.1333 |
| 引用有效率 | 1.0000 | 1.0000 | +0.0000 |
| 平均无依据主张数 | 0.0000 | 0.0000 | +0.0000 |
| 平均上下文字符数 | - | 7,054 | 低于 16,000 上限 |

偏好结果为新候选 7、平局 7、旧方案 1。冻结题目只在候选通过开发集后使用一次。

### AI 知识检索 25 题

| 指标 | 旧生产方案 Top-5 | 新方案 Top-6 | 变化 |
|---|---:|---:|---:|
| Success | 1.0000 | 1.0000 | +0.0000 |
| Recall | 0.9000 | 0.9400 | +0.0400 |
| MRR | 0.7500 | 0.7613 | +0.0113 |
| NDCG | 0.7192 | 0.7688 | +0.0496 |

## 关键产物

- 语料画像：`experiments/chunking_optimization/corpus_profile.json`
- 三轮检索筛选：`experiments/chunking_optimization/retrieval_screening*.json`
- 开发集答案评测：`experiments/chunking_optimization/answer_eval/struct_t205_o50_top6_vs_current/summary.json`
- Holdout 答案评测：`experiments/chunking_optimization/holdout_answer_eval/struct205_o50_ai70_o15_top6_vs_current/summary.json`
- 生产语料：`data/stage4/parent_child/minilm_struct205_o50_ai70_o15_v1`
- 生产 Child 索引：`vector_db/stage4/parent_child/minilm_struct205_o50_ai70_o15_contextual_v1`

## 验收与重建

```powershell
Set-Location E:\RAG
python -m pytest -q
python scripts\experiment\verify_parent_child.py --parent-child-dir data\stage4\parent_child\minilm_struct205_o50_ai70_o15_v1
python scripts\experiment\evaluate_stage4_ai.py --config config\parent_child.json --device cuda --top-k 6 --output-dir experiments\stage4\ai_eval_production
python .\start.py
```

重建语料与 Child 索引：

```powershell
python scripts\data_prep\build_stage4_ai_corpus.py --stage2-dir data\experiments\chunking_optimization\struct_t205_min30_max205_o50 --output-dir data\stage4\parent_child\minilm_struct205_o50_ai70_o15_v1 --child-strategy structured --child-target-tokens 70 --child-min-tokens 30 --child-max-tokens 70 --child-overlap-max-tokens 15 --child-overlap-ratio-cap 0.21 --reuse-raw --offline
python scripts\data_prep\build_vector_db.py --model minilm --chunks-dir data\stage4\parent_child\minilm_struct205_o50_ai70_o15_v1\children --output-dir vector_db\stage4\parent_child\minilm_struct205_o50_ai70_o15_contextual_v1 --text-mode contextual --batch-size 64 --device cuda --overwrite
```

## 回滚

将 `config/parent_child_stage2_reference.json` 覆盖为 `config/parent_child.json`，并将 `config/retrieval_stage1_reference.json` 覆盖为 `config/retrieval.json`。历史索引、评测结果和会话数据库均未删除。
