# R9b 推送与后台测试交接

快照时间：2026-09-25T11:53:50.035979+00:00。**实现主路径已落地，实验与最终验收未完成。**

SV-10/11/12/13/14 的实现已提交；这不代表全部硬门通过。现存失败包括 G-3 的既有 RoPE 静态 FP64（TaskBody 禁改）、G-5 的 BF16 排序、G-7 的部分耗时预算，以及 CG 释放端点和 runtime 保守等待窗口的差异。后续数据可能要求进一步修正；偏离见 [deviations.md](deviations.md)。

| 实验 | 当前状态 |
| --- | --- |
| matrix/llama_s1 | top-3 各 10/10 完成 |
| matrix/llama_s16 | 实测中：7/10 |
| matrix/llama_s4 | top-3 各 10/10 完成 |
| matrix/llama_s64 | 搜索中或由协调器排队 |
| matrix/qwen3_s1 | top-3 各 10/10 完成 |
| matrix/qwen3_s16 | 搜索中或由协调器排队 |
| matrix/qwen3_s4 | 已求解，GPU 排队 |
| matrix/qwen3_s64 | 搜索中或由协调器排队 |
| reference/gqa2_s128 | top-3 各 10/10 完成 |
| reference/gqa2_s4 | top-3 各 10/10 完成 |
| reference/mha4_s4 | 实测中：10/10, 4/10 |

已完成的独立实验：legacy 八格各 10/10；R9 早期信号两格各 10/10；四格 trace 各 10/10 与归因；八格下界核算；490 条拟合与分量回放；分片与逐 tile 一致性；关闭流体模式的逐位对照；两模型纯 home 随机配置各 100 组（G-8 通过）。Llama 带共置的 100 组复核也已完成。

尚待完成且已排队：其余真实模型 top-3 实测、mha4 两格参考测试、Qwen3 带共置的 100 组复核、两个模型 s4 的 A/B/kW 消融、Llama s4 的 S2/3/4、每模型至少 50 个新进程的同步省略检验、两模型七点 θ 网格剩余点，以及剩余赢家的整图访存量审计。后处理由实测 s4 赢家触发，GPU 测量串行并等待外部负载空闲。

当前运行器 PID/启动时刻记录在 [queue_manifest.json](queue_manifest.json)。独立观察器等这些运行器结束后重算全部报告与 verify，不自动提交、不自动推送、不宣称验收通过。进度看 `queued_checks/status.json`，结束时为 `checks_finished_awaiting_review`；完整输出在 `verify_report.log`，检查执行日志在 `queued_checks/report.log`。这只是后台脚本，不会唤醒对话 agent；用户通知后再人工审阅、归档并最终收尾。

189,446,761 字节的原始 trace 以可校验的 gzip 发布，运行中的原文件保持不变。恢复说明见 [trace/README.md](trace/README.md)；未推送证据提交的哈希映射见 `publication_commit_map.tsv`。既有用户改动 `PLACE_EFT2/summary.md` 和 `SYNC_V2/sass_identity/meta.tsv` 不包含在本次推送中。
