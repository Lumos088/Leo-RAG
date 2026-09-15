# 自适应 Context Assembly 冻结 Holdout

- 题目：A057, A037, A027, A035, A062, A007, A002, A023, A038, A016, A036, A066, A006, A061, A052
- 三学科各 5 题；选题在生成答案前按固定哈希冻结

| 指标 | 原 Chunk | Adaptive | 差值 |
|---|---:|---:|---:|
| 必备事实覆盖率 | 0.9211 | 0.9478 | +0.0267 |
| 完整事实答案率 | 0.6667 | 0.8000 | +0.1333 |
| 平均无依据主张数 | 0.0000 | 0.0000 | +0.0000 |
| 引用有效率 | 1.0000 | 1.0000 | +0.0000 |
| 清晰度（1-5） | 4.6667 | 4.8667 | +0.2000 |
| 精炼度（1-5） | 3.8000 | 3.8667 | +0.0667 |

Holdout 通过：**True**

门槛：`{"fact_coverage_not_lower": true, "full_fact_rate_not_lower": true, "citation_validity": true, "unsupported_not_higher": true, "context_budget": true}`

## 逐题结果

- A057 [comparison/parent_rescue]：7/7 → 7/7；context_aware — context_aware 覆盖全部必备事实，引用更充分且结构更清晰精炼，并明确说明资料未涉及的内容。
- A037 [definition/child_highlight_parent]：5/5 → 5/5；baseline — 两个答案均完整覆盖所有必备事实且引用有效，但 baseline 更精炼，冗余度更低。
- A027 [definition/child_highlight_parent]：5/5 → 5/5；context_aware — 两个答案均完整覆盖所有必备事实且引用有效，但 context_aware 表述更精炼，冗余度更低。
- A035 [definition/child_highlight_parent]：3/3 → 3/3；context_aware — 两者均完整覆盖必备事实且引用有效，但 context_aware 更精炼，冗余更少。
- A062 [comparison/parent_rescue]：4/5 → 5/5；context_aware — context_aware 覆盖了全部必备事实（包括分页机制 F04），引用更全面且清晰精炼，而 baseline 遗漏了 F04 且部分内容偏离操作系统语境。
- A007 [enumeration/parent_rescue]：4/4 → 4/4；baseline — 两个答案均完整覆盖四个必备事实且引用有效，但 baseline 更简洁直接，冗余度更低。
- A002 [enumeration/parent_rescue]：6/6 → 6/6；context_aware — 两者均完整覆盖所有必备事实且引用有效，但 context_aware 的引用更精准、表述更精炼，无冗余内容。
- A023 [definition/child_highlight_parent]：7/7 → 7/7；tie — 两个答案均完整覆盖所有必备事实，引用有效且无无依据主张，清晰度与冗余度相当。
- A038 [definition/child_highlight_parent]：4/5 → 4/5；tie — 两个答案均完整覆盖了除F05外的所有必备事实，引用有效且无无依据主张，清晰度和精炼度相当，因此平局。
- A016 [enumeration/parent_rescue]：4/4 → 4/4；baseline — 两个答案均完整覆盖所有必备事实且引用有效，但baseline更简洁直接，冗余度更低。
- A036 [definition/child_highlight_parent]：2/3 → 2/3；baseline — 两个答案均覆盖核心事实且引用有效，但 baseline 更简洁直接，context_aware 包含较多次要细节，冗余度较高。
- A066 [comparison/parent_rescue]：6/6 → 6/6；context_aware — 两个答案均完整覆盖所有必备事实且引用有效，但 context_aware 表述更精炼、冗余度更低。
- A006 [comparison/parent_rescue]：7/7 → 7/7；context_aware — context_aware 覆盖全部必备事实，并额外提供 B 树定义、性能等上下文支持，引用有效，表述更清晰完整。
- A061 [comparison/parent_rescue]：3/4 → 3/4；tie — 两个答案均完整覆盖了所有必备事实，引用有效且无无依据主张，清晰度和精炼度相当，因此并列。
- A052 [definition/child_highlight_parent]：4/5 → 5/5；context_aware — context_aware 覆盖了全部必备事实（包括 F05），引用更充分且结构更清晰，而 baseline 遗漏了 F05。
