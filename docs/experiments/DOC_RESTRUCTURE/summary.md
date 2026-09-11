# DOC_RESTRUCTURE 审计摘要

- 基线：`059f8532509fcf21c9cc96e078903cc1edc13a5a`；当前分支：`tilemega`；HEAD：`8b4a1dc56a13965931bcc5d3cee1c2ab97102737`。
- Prompt SHA256：`0120b63fc5ab425f636e373d3467a82615daff0599a7d93b2526169227f5d73a`。
- 提交：`44816728 docs: move status and todo out of the skeleton`；`b50299bd docs: record structural L2 execution findings`；`3ccb319d docs: add the execution model and plan contract`；`8b4a1dc5 docs: complete execution model restructure`。
- A1–A7：已生成 `audit_lines.tsv`、`audit_numbers.tsv`、`audit_refs.tsv`、`audit_ids.tsv`、`audit_dangling_refs.tsv`、`facts.tsv`；当前审计脚本可一键重跑。A5 代码目录无改动；A6 无新增可确定悬空引用。
- K1–K29 与 P1：✅（依据 prompt 第4节对应代码/实验文件；P1 使用公式计算）。
- 偏离：仓库 HEAD 已高于 prompt 基线，按标题定位；现有历史实现状态保留。未推送远端，因用户未要求发布。
