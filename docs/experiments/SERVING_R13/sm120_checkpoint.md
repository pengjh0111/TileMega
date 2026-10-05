# sm_120 并行验证检查点

这是 Phase C 收尾后的 R13 中间检查点，不是已通过全轮终验的 R13 最终 HEAD。Phase D 在 sm_89 上继续执行；本提交只归档结果、提交生成的配置和提供可携带输入，不改变设备代码、不重启队列。记录拉取后的 `git rev-parse HEAD` 作为 5090 的预验证基线。

## 已确定的输入

verified：Phase C 矩阵完成，三组协议检查均 50/50；完整决定见 `phase_c_retention.json`。保留 C-LP2 的角色分支外提和 C-PG3 的 D=64 KiB、evict_first demand / evict_normal lookahead；C-L2b 与 C-WL 未达到保留门槛，默认关闭。C-AT 的 Qwen3 B16 固定结构优胜臂单列在决定中，联合搜索仍决定最终 attention 几何。允许非分页 L1 循环进入选择，不意味着最终必选循环。

A1 的四格 prefill 均取 PF-R10-noWD。`defaults_r13.json`、`loop_decision.json`、Phase C 决定及生成的 sm_89 配置在本次一起提交；原始 C 对照、HF 检查和新进程证据见 `raw/phase_c_checkpoint/completed_phase_c.tar.xz` 及索引。

## 在 5090 上准备

从仓库根目录执行（仅展开主机几何，不启动 GPU）：

```bash
mkdir -p docs/experiments/SERVING_R13_SM120/inputs
tar -xJf docs/experiments/SERVING_R13/raw/sm120_readiness/geometry_inputs.tar.xz \
  -C docs/experiments/SERVING_R13_SM120/inputs
```

`configs/e2e/llama_r13_sm120.json` 与 `qwen3_r13_sm120.json` 是本次并行配置模板：prefill donor 改为上面的相对路径，cache/output 独立于 sm_89。核对并调整 model.path、test.vllm_python、device.index；在本机重新 export、标定、构建，不复制 sm_89 的 .so 或 target。固定构建只从 donor manifest 取几何/类划分，用 E1 的 target_sm120.json 定价与运行。

E0、E1、E2a 与固定路径预验证可并行；所有计时仍只经 sm_120 的守卫/调度器。本机 guard 空闲功耗重新采样，使用自己的 GPU 锁。现有 R13 jobs/arms 和 builds_r13.py 含 sm_89 绝对路径与 target 快照，不能照搬运行；按 sm_120 prompt 生成本机 queue/env/arms。

## 必须保留的未达标项

- sm_120 PDL trigger=0 的 PTX 位置检查仍失败，尚未判定是检查器误判或实现问题；先定位，首次执行修正按 sm_120 prompt 提交并验证。
- 非分页关闭开关的资源比较仍失败；源码/SASS 相同，不把资源失败改记通过。
- Llama B1 B0h 的 C-1 失败；5090 仍需独立 C-1 检查，不能把它直接当成正确的默认候选。
- sm_89 Phase D、最终配置/HEAD 和跨架构报告尚未结束。正式终版矩阵需对齐最终共同源码/配置；本次并行属于已记录的中间基线验证。

在 5090 使用独立分支，例如 `git switch -c r13-sm120-validation`；首次执行修正不要直接改写 sm_89 正在测量的 worktree。

## 2026-10-05 sm_89 终验回放更新

Phase C 保留决定不变，Phase D 原构建和四格终版 C-1/C-2 已结束；还需两轮预注册金丝雀替代测量后才能称最终封版。设备代码仍为 dedd849f3 的冻结状态。本次新增的是 CPU 验收工具修正：原 sm_120 trigger=0 位置失败是 ret 后不可达冷块误判，四项 PTX 控制流检查均通过；同名 callee 资源需按入口调用上下文区分，已有内核源码/资源/SASS 比较均通过。详见 raw/final_review。

5090 工作树应同步 ptx_pdl_check.py、compare_kernels.py、arch_checks.py、anchor.py 与精确 batch 路径匹配的分析工具，避免沿用误判。该更新不把本机编译通过当作 sm_120 的性能/同步正确性；5090 的硬件验证仍须执行。本文早期“位置/资源尚未解决”的记录保留为历史，以此更新为准。
