# DOC_RESTRUCTURE 审计摘要（v2.1 重整与修复）

- 基线：`059f8532`。首次重整提交为 `44816728`…`83da9fe6`；修复提交为 `28354094`…`361a515a`，以及添加本目录的提交。
- 审计时 HEAD 为 `361a515a`，即文档最终状态；本目录的提交只新增审计文件。重跑方式：`python3 docs/experiments/DOC_RESTRUCTURE/audit.py`，非零退出即失败。
- 修复所用 prompt 的 SHA256 前缀为 `663e7186f99c841a`。首次记录的 `0120b63f…` 与此不同；若执行时改动过 prompt，以改动后的 prompt 为准复核。

## 首次重整的问题（已由本次修复取代）

**审计无效。** 首次提交中的 `audit.py` 是占位实现：
- A1 只检查几个标题是否存在；
- A2–A4 直接写 PASS；
- A7 对 K1–K29 直接写 ✅；
- A6 输出为空。

因此首次报告的全部审计结论无效。

**内容未落地。**
- skeleton 的 §7（833 行）未迁出。
- §5.7 只有一句话，且位于 §5.6 之前。
- S-5、S-7、S-8、S-10、S-12、S-14、S-16(a)、S-17、S-18 均未落地。
- STATUS 混入了 skeleton 的 `# 2.` 标题，§1.5.2 出现两次，§1.5.4 位置错误。
- TODO 的 §2、§3 为空，EX 详述挂在它们之后，文件中复制了整份 v2.0 §7，末尾还有重复的 EX 残段。
- F-126–F-130 被压缩为一两句话，F-82/F-86/F-118 的注记缺失。
- `PROPOSED_SKELETON_CHANGES.md` 未修改。
- 提交信息与内容不符（`3ccb319d` 只含审计文件）。

## 修复方式

`build.py` 从基线出发，按 prompt 第 6 节逐项确定性重建全部文档；需原样写入的文本直接从 prompt 文件中抽取。重跑方式：
`TILEMEGA_RESTRUCTURE_PROMPT=<prompt> python3 docs/experiments/DOC_RESTRUCTURE/build.py full`
重跑结果与已提交文档逐字节一致。

## 审计结果（`audit_summary.txt`）

```
A1	PASS	2044 non-empty base lines; missing 0; annotated 6
A2	PASS	253 numeric tokens in new lines; unsourced 0 (review audit_numbers.tsv)
A3	PASS	182 references; unresolved 0
A4	PASS	34 P ids, 14 EX headings; problems 0
A5	PASS	code changes 0; other experiment changes 0
A6	PASS	6 code references to sections absent from the skeleton (pre-existing; report only)
A7	PASS	30 facts; failing []
```

## 与 prompt 的偏离

1. **F-131 未写入。** 未确认 `E2E_L2/raw/part3/gqa2.cu` 与 L2_ATTRIB 为同一 BF16 fixture。STATUS G12 的证据列相应写为"⚠️ inferred（权重字节未核实）"。
2. **§3 索引保留通配路径。** 如 `docs/experiments/P3/derived-*.md`。
3. **archive 头部日期** 记为 2026-09-12。

## 需要人工决定

- A6 列出的 6 处代码注释引用了 skeleton 中不存在的 `§0`、`§0.1`、`§0.3`（历史编号，见 `audit_dangling_refs.tsv`）。按 H1 未修改，映射待定。
- STATUS §1.5 原文第 331 行引用的 `docs/experiments/E2E_GEN/generated_e2e` 在仓库中不存在，属于原有问题。
