# R13 sm_120 并行启动核对

核对的本地代码 HEAD：`2241cbabe9710dd957fa16651d96945756e06e4e`。本地 `origin/tilemega` 引用仍为 `48c013d18`；未联网刷新远端。本核对不启动 GPU 任务、不改变 sm_89 队列。

规格：[/root/Prompt/TileMega_R13_sm120_prompt.md](/root/Prompt/TileMega_R13_sm120_prompt.md)，SHA256 `84522ced9aa38003ddd87b7811804f202a99a936df9e1ae37b45c35bab22e286`。开头明确要求 sm_89 R13 完成并推送，以最终 HEAD 为基线；目前不满足，提前执行必须声明为准备/预验证，不能称为正式完成该规格。

| sm_120 项 | 当前依赖与可并行范围 |
|---|---|
| E0 环境、Caps、单元测试 | 可独立推进；本机实测属性、CUDA/Python/模型指纹，不能沿用 sm_89 环境读数 |
| E1 全节标定、TL-2 | 工具已实现，可针对本机生成 target；源码冻结印章变化时重测失效分节 |
| E2a MB-1a–f | bulk/cluster/PDL 臂已实现并编译；可首次执行，但所有计时经本机调度器/守卫 |
| E3 固定几何、E2b/E2c | B0/B0h/PR/PS/PSA、循环、split-K LA、trace 的代码与几何输入已具备；可先做预验证。PDL trigger=0 尚有位置检查失败，先诊断，再做该臂性能/同步结论 |
| PF-120 | A1 选择已有：四格均 PF-R10-noWD；默认值与原始几何原先仅在本地，现在打包，可在本机重建 |
| R13F-120 联合选择 | SL-5 已实现；C 保留/回退、L1 loop 默认资格及终版配置尚未生成，正式构建待同步 sm_89 最终结果 |
| E4/E5 正式矩阵、跨架构 E6 | 固定机制的试运行可以先做；正式结果须固定一致编译器版本，R13 T9/T10 和最终 HEAD 未完成，跨架构最终比较待齐 |

已实现的关键入口：`ArchDispatch.h` 的 Sm120 Caps；`ServingRuntime.cuh` 的循环能力/launch_steps；`ModelHarness.cuh` 的非分页 PDL/L1 loop；`ServingPages.cuh` 的 bulk 页装载；`SkeletonSearch.cpp` 的两个种子；`cli.py` 两级选择；loadbench 的 bulk、cluster、PDL 臂。存在代码不等于对应硬件正确性已验证。

已知未达标：sm_120 trigger=0 的 PTX 检查报告 publication follows late trigger，尚未判明是控制流检查误判还是实现问题；trigger=1 检查通过，仍须本机验证。非分页参考资源比较失败但源码/SASS 相同；Llama B1 B0h 的 C-1 未通过。它们须保留在新环境报告中，不能把 sm_89 的成功检查替代 sm_120 的同步检验。原始记录见 `raw/inspection_05/completed_C_evidence.tar.xz`。

## 可携带的输入

`raw/sm120_readiness/geometry_inputs.tar.xz` 包含 donor manifest、同 DN 类划分、分页原种子及 A1 prefill 决定；`inputs_manifest.json` 记录每项原路径、SHA256 与 checkpoint。不包含 .so/.cu，也不携带可供 sm_120 求解的 sm_89 target。解包后用本机 export、E1 target 和编译器重新构建；donor manifest 只提供几何/类划分，不能直接作为本机可运行计划。

R13 的现有 `builds_r13.py` / `fixed_builds.py` 写死 `/root/r13_work/target_r12b.json`，jobs/arms 含本机缓存路径，配置含本机模型/Python 路径。不能原样运行 sm_89 队列。sm_120 需要自己的 queue/arms/env、target_sm120.json、空闲功耗及锁；这些是规格要求的新环境编排，不是缺失的设备功能。

## 并行方式

在 5090 上单独分支、固定 checkpoint、独立输出目录与 cache，先推进 E0/E1/E2a，再预验证固定几何路径。这里继续完成 sm_89 C/D，不把 5090 首次执行修正直接写入正在测量的 sm_89 worktree。双方最终同步已保留的开关和修正，固定最终共同基线，受影响的构建/标定再更新，随后进行正式 R13F-120 与跨架构比较。

建议并行补充条款：允许从记录的 R13 checkpoint 提前执行环境准备、标定、微基准与固定机制预验证；R13F-120、终版对比与报告以最终 R13 HEAD 对齐。若不作这项补充，按原规格应等待 R13 完成后正式开工。
