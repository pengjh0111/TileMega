# R9b 测试停止与部分结果归档

✅ verified：2026-09-25，`ps` 核查未发现 R9b 搜索、验证或排队测试进程。`queued_checks/status.json` 的 `checks_finished_awaiting_review` 表示检查程序退出，不表示全部实验完成或全部门通过。用户指示未做的测试不再测试；本次没有重新启动测试。原始自动检查输出保留在 `verify_report.log`，完整逐门报告保留在 `summary.md`。

## 已完成并可报告的结果

| 范围 | 已完成结果 | 原始证据 |
| --- | --- | --- |
| 真实模型 legacy 对照 | 两模型 × seq 1/4/16/64，八格各 10 个新进程，内部逐位一致 | `controls/`、`report_tables/performance.tsv` |
| 新管线真实模型 | Llama s1：4.446912 ms，`L2/T_floor=1.765767`，为同会话 legacy 的 0.820095；Llama s4：4.446888 ms，1.765129，0.686812；Qwen3 s1：8.041976 ms，2.293451，0.975286。每格 top-3 各 10 次，选出赢家 | `matrix/{llama_s1,llama_s4,qwen3_s1}/`、`report_tables/performance.tsv` |
| Level 1 / 流体模拟排序 | 纯 home：两模型各 100 组，Spearman 0.994287 / 0.990735；带共置：各 100 组，0.994671 / 0.991023。后者的 Qwen3 100 组在此次归档中补齐 | `validation{,_colocated}/{llama,qwen3}/samples.tsv`、`report_tables/consistency.tsv` |
| θ 网格 | Llama seq 1/2/4/8/16 五点、Qwen3 seq 1/2/4 三点完成；此次补齐 Llama s16 与 Qwen3 s4 | `theta/*/completed.tsv`、`report_tables/theta.tsv` |
| Llama s4 纯模板消融 | 10/10，L2 4.444592 ms，`L2/T_floor=1.764218`，移离 home 为 0/14525 | `ablations/llama_s4/template/`、`report_tables/additional_measurements.tsv` |
| 参考模型 | gqa2 s4/s128 的 top-3 各 10 次完成，赢家 L2 0.124928/0.218568 ms | `reference/gqa2_s{4,128}/` |

这些数据只支持已完成格的结论。模型比例以本轮同会话 legacy 为分母。以上新增结果由停止前已经运行的进程产生，本次仅整理、归档和推送。

## 停止时未完成的测试

- 真实模型新管线：Llama s16 求解完成，但 top-3 只完成前两个候选各 10 次；Qwen3 s4 求解完成，top-3 尚未实测；Llama s64、Qwen3 s16/s64 未完成求解。因此完整矩阵仅 **3/8** 格可选赢家，G-9 与 G-10 **无法判定**，不可把现有三格外推成 6/8 或六格几何平均。
- 参考模型：mha4 s4 完成 top-3 的 10/10、10/10、8/10；mha4 s128 未完成。G-2 所需参考模型与每模型 ≥50 个共置省略同步新进程未齐；后者两个模型均为 0/50。不得宣称省略等待的正确性门通过。
- θ 网格为 8/14 点；Llama s32 与 Qwen3 s8 留有未完成搜索前缀，余下点未完成。不能从部分前缀推断完整区间。
- 消融仅 Llama s4 纯模板完整；其 EFT 为 3/10，k=W 未测；Qwen3 s4 三臂与 Llama stages 2/3/4 未测。真实模型流量审计的 skeleton 五格、完整 top-M 物化的部分格也未齐。

`report_tables/incomplete.json` 和 `verify_report.log` 逐路径记录缺项。未完成实验按用户本次指示停止，不将其改记为 PASS。

## 已知不依赖缺项的未达门

✅ verified：G-3/K-12 的 FP64 静态指令数非零，来自本轮禁止修改的既有 RoPE 路径；G-5 的 BF16 回放 Spearman 为 0.83756/0.84262，低于既定门；G-7 的预热流模型最慢 25.665 ms，已完成求解格也超过 30 min 预算；runtime 释放窗抽样有 472/2064 个端点差异。这些是实质性的未达项，不由停止测试造成。G-8 的纯 home 排序门已通过；其余缺样本门保持未验收。

为避免推送数 GB 的可重建中间产物，版本库保留命令、退出码、逐进程日志、样本表、求解摘要与必要的 CG/流模型指标；大型逐边/逐 task dump、ELF 在本地原目录保留，未据此宣称完整证据已推送。两个用户原有的未提交改动 `docs/experiments/PLACE_EFT2/summary.md`、`docs/experiments/SYNC_V2/sass_identity/meta.tsv` 未纳入本次提交。
