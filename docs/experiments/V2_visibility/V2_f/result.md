# V2-f：剥掉 tile 尺寸与 reduce

## 结论（先行）

✅ **单元素也好、整块搬运也好，只要没有跨 warp 归约就永不出错；只要有跨 warp 归约就出错。
`reduce` 的共享内存级是唯一的开关，与"读到的数据是谁写的"完全无关。**

## 1. 逐步剥离

grid=340，每格 30–50 次全新进程。

| 变体 | 消费者做什么 | 有跨 warp 归约？ | 失败率 |
|---|---|---|---|
| `v2_f1` | 自旋后读 **1 个元素**（`tile<1xf32>`），不 reduce | 否 | **0/50** |
| `v2_f1024` | 读 `tile<1024xf32>`，**逐元素**全写回 out，不 reduce | 否 | **0/50** |
| `v2_f_const` | 自旋后 reduce 一个 `broadcast(1.0)`，**不读内存** | 是 | **0/30** |
| `v2_f_hostdata` | 自旋后 reduce chunk 200，该 chunk 由 **host 在 launch 前写好**，无任何 producer 触碰 | 是 | **36/50 (grid 170)**，**50/50 (grid 340)** |
| `v1_min` | 自旋后 reduce chunk 0（producer 写） | 是 | 50/50 (grid 340) |

### `v2_f_hostdata` 是关键对照

它的数据是 **host 在 kernel 启动之前 `cuMemcpyHtoD` 写进去的**，kernel 里没有任何 block
写过 chunk 200。跨 block 数据可见性在这里**在物理上不可能**是解释。它照样失败，
错值签名（512/768）与 `v1_min` 一模一样。

✅ **已验证：这条单独就足以把"跨 block 数据可见性"从病因里排除。**

### `v2_f_const` 通过，是我一个假设被自己推翻的地方

看到 `v2_f1024`（无 reduce）通过后，我提出"自旋的发散退出打乱了 CTA 的 barrier 配对，
破坏共享内存归约"。`v2_f_const`（有自旋、有完整跨 warp 归约、但不读内存）却 30/30 全过，
**并且 SASS 确认那一级 `STS`/`BAR.SYNC`/`LDS` 没有被折叠掉**。

这削弱了该假设，我一度停止断言它。后来的尺寸扫描（§2）又把它救回来了：
差别在于 `v2_f_const` 没有 load 延迟，warp 之间拉不开足够的时间差。
⚠️ **这一条至今没有独立坐实**，是本轮遗留的最大不确定点。

## 2. tile 尺寸扫描（修正后）

grid=340，30 次全新进程，数据一律由 host 预填（`V2_HOSTFILL_CHUNK=128`），
消费者 reduce 一个 `tile<Nxf32>`。

```
tileN  expect  failed/30   bad-value histogram
32     32      0/30
64     64      30/30       32  x205
128    128     30/30       64  x455
256    256     30/30       128 x493
512    512     30/30       256 x479
1024   1024    30/30       512 x422
```

✅ **错值恒等于正确答案的一半**，且 N=32（单 warp）从不出错。

### SASS 上的结构判据

```
v2_hd32   : LDG -> BAR.SYNC -> SHFL.BFLY x5 -> @!P0 STG.E
            没有 STS，没有 LDS，没有跨 warp 级             -> 通过
v2_hd1024 : LDG x8 -> BAR.SYNC -> SHFL.BFLY x5
            -> @!P0 STS -> BAR.SYNC -> @!P1 LDS -> SHFL x2
            -> @!P0 STS -> BAR.SYNC -> LDS -> @!P0 STG.E   -> 失败
```

✅ **迄今为止所有 V1/V2 变体中，失败集合恰好等于"归约里有共享内存跨 warp 级"的集合。**

### 这证伪了 §0.4(2) 的数值巧合

§0.4(2) 认为 128 个元素 = 一条 `STG.E` 的份额。但 N=64 时丢的是 **32**，
既不是 128，也不是任何一条 store 指令的份额。真正的量子是 **warp partial**
（N=1024 时 = 256 元素，N=64 时 = 32 元素），不是访存指令。

## 3. 我自己制造并作废的一批数据（方法论记录）

第一次做尺寸扫描时，我用 sed 从 `v2_hd1024` 派生 `v2_hd128`/`v2_hd256`，
只改了 partition 的 tile 宽度，**没改索引**。于是 `tile=(128)` 时读的是元素
`200*128 = 25600`，落在 `V2_HOSTFILL_CHUNK=128` 边界（元素 131072）**之下**，
读到的是 poison。结果直方图出现"期望值 128、错值也是 128"的自相矛盾。

我据此发现问题、核对 MLIR 确认、**宣布那两个数据点作废**，改用 `gen_hd.py`
把每个 N 的读取基址统一钉死在元素 204800（`idx = 204800/N`）。上面 §2 的表是重做后的。

## 复现

```bash
cd /data/tilemega/docs/experiments/V2_visibility
V2_HOSTFILL_CHUNK=128 V2_REPS=1 ./v2_host V2_f/v2_f_hostdata.cubin v2_f_hostdata 340 scalar 1024 1
python3 V2_f/gen_hd.py       # 重新生成并编译整个尺寸扫描
```
