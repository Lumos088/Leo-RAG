# Stage 2 端到端答案质量对比

- 题目数：15（三学科各 5 题）
- 条件：相同 DeepSeek 模型、提示词、Top-5、上下文字符预算
- 说明：LLM 自动评审用于低成本筛查，重要结论仍建议人工抽检。

| 指标 | 原 Chunk 策略 | Parent RRF | 差值 |
|---|---:|---:|---:|
| 必备事实覆盖率 | 0.8778 | 0.8461 | -0.0317 |
| 完整事实答案率 | 0.6667 | 0.5333 | -0.1333 |
| 平均无依据主张数 | 0.2000 | 0.0667 | -0.1333 |
| 引用有效率 | 1.0000 | 1.0000 | +0.0000 |
| 清晰度（1-5） | 4.8667 | 4.8000 | -0.0667 |
| 精炼度（1-5） | 4.1333 | 4.3333 | +0.2000 |

偏好计数：`{"parent_rrf": 4, "tie": 7, "baseline": 4}`

## 逐题结果

- A004（Operating System）：parent_rrf — 两个答案都覆盖了主要必备事实，但parent_rrf的引用更准确、无越界引用，且未引入实时系统调度目标等无关内容，整体更精炼。
- A008（Operating System）：tie — 两个答案均完整覆盖所有必备事实，无无依据主张，引用编号均有效，清晰度和精炼度相同。
- A001（Operating System）：tie — 两个答案都完整、准确地覆盖了四个必要条件，且引用均有效；baseline 额外引用了同样支持相关条件的 [2]，parent_rrf 仅引用 [1]，两者在事实完整性和依据充分性上相当。
- A003（Operating System）：parent_rrf — 两者均完整覆盖全部必备事实且引用有效，但 parent_rrf 表述更精炼、冗余更少。
- A009（Operating System）：tie — 两个答案均完整覆盖三项必备事实、无无依据主张且引用有效，清晰度与精炼度相当。
- A011（Computer Network）：parent_rrf — parent_rrf 的答案事实依据更充分且无无依据主张，而 baseline 包含上下文不支持的“面向连接的通信”功能。
- A054（Computer Network）：parent_rrf — parent_rrf 覆盖了更多必备事实（包括 TCP/UDP 首部开销和连接时延），且引用有效、表述精炼。
- A020（Computer Network）：baseline — 两个答案都只覆盖了 F01，但 baseline 更简洁清晰，且未引入与问题无关的额外引用内容。
- A024（Computer Network）：tie — 两个答案均完整覆盖四项必备事实，引用有效且无无依据主张，清晰度与精炼度相当。
- A058（Computer Network）：baseline — 两个答案均完整覆盖必备事实且引用有效，但基线答案更聚焦于数据通信中同步与异步传输的核心区别，未引入操作系统语境等次要信息，依据更集中。
- A010（Data Structure）：tie — 两个答案均完整覆盖两种方法，引用有效且表述清晰，无实质差异。
- A033（Data Structure）：baseline — baseline 覆盖了全部必备事实（包括 Kruskal 应用和序列实现复杂度），引用有效且表述清晰；parent_rrf 缺少 F03 和 F04，事实完整性不足。
- A039（Data Structure）：tie — 两个答案均覆盖了核心事实、引用有效且清晰精炼，baseline 额外提及选举问题但未增加事实覆盖，整体质量相当。
- A059（Data Structure）：tie — 两个答案均完整覆盖全部必备事实，引用有效且无无依据主张，清晰度与精炼度相当。
- A063（Data Structure）：baseline — baseline 覆盖全部必备事实，引用有效且依据充分，并补充了适用场景，事实完整性优于 parent_rrf。
