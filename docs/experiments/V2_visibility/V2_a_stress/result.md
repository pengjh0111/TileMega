# V2-a′：V1-a 是机理安全，还是时序侥幸？

## 结论

✅ **机理安全——但不是 V1 认为的那个机理。**

grid=340，每格 **500** 次全新进程（V1 只跑过 50 次）：

```
cell                       failed/500
plain                      0/500
host delay 5000us          0/500
host delay 50us            0/500
non-blocking stream        0/500
no-spin (flag preset)      0/500
------------------------------------
合计                        0/2500
```

四种扰动（长延迟、短延迟、换非阻塞 stream 取代 `cuStreamSynchronize`、以及完全不自旋）
一次都没有翻车。V1-a 的干净纪录不是 50 次的运气。

## 但 V1 对"为什么干净"的归因是错的

V1 认为 V1-a 通过是因为**每个 block 读自己独占的 chunk**。真实原因是
**V1-a 根本没有 `reduce`**：

```
$ grep -c reduce V1_a/v1_a.mlir
0
$ cuobjdump -sass V1_a/v1_a.cubin | grep -c '\bSTS\b'   # 1
$ cuobjdump -sass V1_a/v1_a.cubin | grep -c '\bLDS\b'   # 0
```

V1-a 的消费者把 `tile<1024xf32>` 乘 2 后整块存回 out，**没有任何跨 warp 归约级**。

直接的 A/B（见 `V2_e/result.md` 的 2×2）：把 V1-a 的形态原样保留、
**只加上一个 `reduce`**（`v2_y_shared_excl`：单生产者、共享 flag、消费者读独占 chunk bx
然后归约），grid=340 → **50/50 失败**。独占数据没有救它。

✅ **已验证：`reduce` 是开关，独占数据不是。**

## 复现
```bash
N=500 ./V2_a_stress/run.sh
```
