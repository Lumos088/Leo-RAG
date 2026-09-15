# Stage 2 Parent-Child RAG 实施与验收报告

## 1. 上线结论

Stage 2 已通过开发集与冻结 Holdout 验收，并接入 Web API 作为默认检索策略。原 Stage 1 检索链路仍保留，可通过 `config/parent_child.json` 的 `enabled` 字段立即回退。

这里的“上线”指接入 `E:\RAG` 本地应用，不包含公网部署或 Git 推送。

## 2. 最终方案

```text
用户问题
  → Child Dense + BM25 召回
  → RRF 融合
  → BGE Child 重排
  → Child 分数聚合到 Parent
  → Parent BGE/RRF 重排
  → 原 Stage 1 BGE 结果补入一个未重复 Parent
  → 问题类型路由
       definition：关键 Child + 完整 Parent
       enumeration/comparison：完整 Parent
  → 统一引用与 16000 字符预算
  → DeepSeek 生成
```

核心分工：Child 提供精确匹配信号，Parent 提供完整语义范围并承担引用归属。补位机制降低单一 Parent-Child 排名遗漏关键证据的风险。

## 3. 关键参数

| 参数 | 最终值 |
|---|---:|
| Child 目标长度 | 200 tokens |
| Child overlap | 50 tokens |
| Dense 候选 | 75 |
| BM25 候选 | 50 |
| RRF k | 20 |
| 重排候选 | 50 |
| Parent 重排候选 | 20 |
| Parent 主 Top-K | 5 |
| BGE 补位 | 1 个未重复 Parent |
| 定义题 Child 高亮 | 每 Parent 1 个 |
| 最大上下文 | 16000 字符 |

## 4. 实验过程与淘汰原因

| 策略 | 必备事实覆盖率 | 完整事实答案率 | 结论 |
|---|---:|---:|---|
| 原 Chunk 开发集基线 | 87.78% | 66.67% | 对照组 |
| Parent RRF Top-5 | 84.61% | 53.33% | 完整性下降，淘汰 |
| Parent RRF + 1 补位 | 88.11% | 66.67% | 覆盖恢复，但完整率未提升 |
| Child 高亮 + Parent | 86.78% | 66.67% | 局部强化导致其他事实被忽略 |
| Parent 路由 + Child-only Top-2 | 79.28% | 46.67% | 上下文不足，淘汰 |
| 最终自适应策略 | 89.44% | 73.33% | 开发集通过 |

单纯按 Parent 去重会扩大覆盖广度，但可能挤出关键 Parent；单纯把 Child 交给 LLM 又会丢失跨片段信息。最终方案根据问题意图决定是否突出 Child，同时保留 Parent 和一个 Stage 1 补位证据。

## 5. 开发集验收

15 题，操作系统、计算机网络、数据结构各 5 题。

| 指标 | 原 Chunk | 最终策略 |
|---|---:|---:|
| 必备事实覆盖率 | 87.78% | 89.44% |
| 完整事实答案率 | 66.67% | 73.33% |
| 引用有效率 | 100% | 100% |
| 平均无依据主张数 | 0 | 0 |
| 清晰度 | 4.60 | 4.87 |
| 精炼度 | 3.93 | 4.13 |

关键题 A033、A054、A063 分别达到 5/5、8/8、5/5。

## 6. 冻结 Holdout 验收

Holdout 在生成答案前通过固定哈希选择，排除开发集，三学科各 5 题。

| 指标 | 原 Chunk | 最终策略 |
|---|---:|---:|
| 必备事实覆盖率 | 92.11% | 94.78% |
| 完整事实答案率 | 66.67% | 80.00% |
| 引用有效率 | 100% | 100% |
| 平均无依据主张数 | 0 | 0 |
| 清晰度 | 4.67 | 4.87 |
| 精炼度 | 3.80 | 3.87 |
| 平均上下文字符数 | 4148 | 5974 |
| 平均生成耗时 | 2.432 秒 | 2.514 秒 |

最终策略事实覆盖率、完整答案率均高于 Holdout 基线；生成耗时增加约 3.4%，上下文仍低于 16000 字符预算。

## 7. 代码与配置改动

- `scripts/pipeline/parent_child_retrieval.py`
  - 支持 Parent BGE/RRF 重排。
  - 支持复用同一 BGE reranker，避免重复加载交叉编码器。
  - 支持 Parent 内 Child 高亮选择。
  - 关闭前端 rerank 时不再隐式执行 Parent rerank。
- `scripts/pipeline/context_assembly.py`
  - 问题类型路由。
  - Stage 1 BGE 证据补位。
  - Parent 引用归属、Child 高亮和字符预算。
- `web_ui/api/rag_service.py`
  - 非流式与流式接口默认接入最终策略。
  - 保留原检索回退路径。
  - 流程事件与现有前端 Pipeline 同步。
- `config/parent_child.json`
  - 默认启用最终策略及全部已验证参数。
- `config/retrieval.json`
  - 默认 Top-K 调整为 5，与评测条件一致。
- `tests/test_parent_child_runtime.py`
  - 覆盖路由、补位、引用、配置与开关兼容性。

## 8. 验收命令

```powershell
cd E:\RAG

# 单元测试
python -m pytest .\tests\test_parent_child_runtime.py -q

# Parent-Child 数据完整性
python .\scripts\experiment\verify_parent_child.py

# 开发集结果
Get-Content .\experiments\stage2\answer_eval\adaptive_definition_highlight_vs_baseline\summary.md

# 冻结 Holdout 结果
Get-Content .\experiments\stage2\answer_eval\adaptive_holdout15_v1\summary.md

# 一键启动
python .\start.py
```

## 9. 回退方法

将 `config/parent_child.json` 中：

```json
"enabled": true
```

改为：

```json
"enabled": false
```

然后重启 FastAPI 与 Streamlit。系统将恢复 `config/retrieval.json` 指向的 Stage 1 检索链路，无需删除 Stage 2 数据或索引。

## 10. 评测边界

- 端到端事实与引用指标由 DeepSeek 自动评审，适合低成本回归筛查，但不能完全替代人工专家评审。
- 当前题库覆盖三门课程的 60 个可评问题；冻结 Holdout 仅使用其中 15 题。
- 后续扩充 AI、机器学习、深度学习资料时，应新增独立标注集，不能直接沿用本轮阈值宣布新领域通过。

## 11. 最终回归记录

- Parent-Child 数据校验：PASS。
  - 7587 个 Parent 全部保留原 ID。
  - 20630 个 Child 全部能映射到 Parent。
  - 无 Parent 缺少 Child。
  - Child 最大 215 tokens，低于 400 tokens 硬上限。
- 运行时单元测试：`5 passed`。
- 60 个 eligible 问题的规则路由与评测标签：`0` 个不一致。
- 完整流式链路：PASS。
  - progress 事件依次覆盖 Query、Dense、BM25、Fusion、BGE、Context、LLM、Answer。
  - 并查集测试返回 6 个带 Parent 引用的结果，并成功生成最终答案。
- `python start.py --no-browser`：PASS。
  - FastAPI `GET /api/health` 返回 200。
  - Streamlit 在 8501 端口正常启动。
  - 验收后进程已关闭，8000/8501 端口均已释放。

## 12. 传统检索指标复评

最终生产策略为 Parent RRF Top-5 加 1 个 BGE 补位 Parent，因此实际 LLM 上下文深度为 6。传统 qrels 指标与原 Chunk 的结果如下：

| 指标 | 原 Chunk | Stage 2 Production |
|---|---:|---:|
| MRR@10 | 0.9833 | 0.9611 |
| NDCG@5 | 0.8083 | 0.7639 |
| NDCG@6 | 0.8058 | 0.7675 |
| Recall@5 | 0.3312 | 0.3067 |
| Recall@6 | 0.3761 | 0.3606 |
| Success@1 | 0.9667 | 0.9333 |
| Success@5 | 1.0000 | 1.0000 |
| 上下文事实覆盖@6 | 0.8959 | 0.8987 |
| 上下文事实完整率@6 | 0.6667 | 0.7000 |

结论：传统文档级排序指标出现小幅回落，但实际 6 条上下文的事实覆盖和完整率提高，且开发集、冻结 Holdout 的最终答案质量提高。Stage 2 属于以答案充分性为目标的取舍，不能表述为所有检索指标全面超过基线。由于生产仅输出 6 个 Parent，NDCG@10 会因缺少第 7～10 名而受到额外惩罚，不应作为这一版的主要横向比较指标；应优先比较相同深度的 NDCG@5/@6。
