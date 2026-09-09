# T5.1：isl 引用与显式生命周期

✅ 基线 c8be09e。先加入 scoped 引用审计、保留泄漏，运行参数化工具在首个
`CouplingRelation::Points` 出口即拒绝：`before=0 after=4`，exit=134，见 before.txt。
不是仅仅静态析构顺序：CollectPoint 每个点调用两次 __isl_give 的
isl_point_get_space，均未 free。两个枚举点已足够留下四个引用。

修复使用 Point/Space RAII；出口异常同样释放。QuasiPolynomial 的 parse、
SubstituteParams、Eval 及新增 CG 输入/绑定路径也保留 scoped audit。
审计直接读取匹配已链接 isl 的私有头文件定义，不猜内存布局；该依赖限于
ISLContext.cpp，公共头文件不暴露 isl 内部结构。

工具与测试显式在 main 中持有 IslContext，晚于其创建的 MLIR/isl 值先销毁。
SharedIslContext 只借用当前线程调用者拥有的对象，无静态 owning context；
无调用者时明确报错。支持嵌套作用域，退出恢复外层。析构要求引用为零，
TILEMEGA_ISL_AUDIT=1 打印实际剩余计数，不将 exit=0 代替零残留证明。

✅ after.txt：两模型输入位模式仍 2154/2154，两个有限域 DP 对照不变，
`ISL_CONTEXT remaining=0`；完整 24/24 CTest 通过。这里是 CPU 所有权验证，
不是 GPU 同步/竞态的 50 进程验收。

代码：lib/Analysis/CouplingRelation.cpp（CollectPoint/Points），
lib/Analysis/ISLContext.cpp（ReferenceCount、IslReferenceAudit、析构），
tools/tilemega-parametric.cpp（main 的调用者作用域）。
