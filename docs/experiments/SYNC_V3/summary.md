# TileMega R4：目标同步协议完成与剩余差距

## 1. 基线、提示词与提交顺序

基线：`ee905036d2ec0c9dc880df604097981552423e53`。分支：`tilemega`。提示词保存在仓库外 `/root/Prompt/TileMega_R4_prompt.md`，SHA256：`669c77165cc4f776088b37f24fda2039f8c2b29ae49f2a3e3133cd000d35484c`。

恢复后的严格依赖顺序为 A `23c5d618` → target 单独冻结 `03053089` → B 五臂数据 `204df980` → C1 `e6dd44bc`。这三处先决顺序成立。必须保留的历史偏离：早先 `66ae000e` 曾在 B 数据完成前提交 C1，随后 `8299ccaa` 回退；该历史违规没有被抹去。恢复阶段依用户最新指令完成局部修复，不能把原始历史称为完全满足 H4。

最终 SASS 证据采用可审计的父子提交：在全部代码/文档提交完成后编译，`sass_identity/manifest.json` 记录被编译的 HEAD；最后一个提交只携带该目录的证据，验证器要求当前 HEAD 等于被测提交，或是其仅修改证据的直接子提交。提交无法在自身内容中存储自身 hash；最终两次提交的实际 hash 见交付消息与 `git log`。

```text
62aacadf trace: rebuild predecessors from the task graph
d549c5a8 experiments: freeze the per candidate ceiling targets
79fe7549 experiments: price the device scope release fence
66ae000e runtime: release once per CTA after the barrier
8299ccaa runtime: defer C1 until the prerequisite measurements pass
23c5d618 trace: recover elided dependencies and partition causal intervals
36cc8c9b experiments: audit the round four protocol prerequisites
623d0b74 docs: record the round four prerequisite stop
fbd27a9f experiments: stamp the sass identity at head
03053089 experiments: freeze the per candidate ceiling targets
204df980 experiments: price the device scope release fence
e6dd44bc runtime: release once per CTA after the barrier
a1eb8dd7 docs: unseal the single writer release rule
75e47208 runtime: publish asynchronously from one warp
ee0ce4c8 runtime: satisfy in-CTA dependencies in shared memory
baaebedf runtime: arrive at cluster scope where available
f026dfe6 runtime: carry shard closure through reduction publication
b8d66aa9 experiments: remove window polling from the no-wait probe
3c98f7f0 experiments: record the registered protocol gate across configurations
bdbf3217 experiments: validate the completed sync protocol
1251e784 experiments: measure the completed sync protocol
e6672359 solver: extend a chain only when the hop pays
884e8292 experiments: remeasure the cost aware chain
36c82ea8 experiments: add the sm_120 runners for round four
cd14eef5 experiments: complete the real width protocol ablation
```

## 2. 逐门结果与完整自检输出

| 门 | 类型 | 结果 | 原始证据重算结果 |
|---|---|---|---|
| H2 | hard | PASS | gqa2=6802641 bytes; mha4=6802641 bytes; input digests match; tested HEAD or artifact-only direct child=True |
| A-a | hard | PASS | 32/32 floors <= measured |
| A-b | hard | PASS | 32/32 split closes, nonnegative, wait <= span |
| A-c | hard | PASS | chain: 20 nodes 242688 ns; rotate: 20 nodes 242688 ns |
| A-d | report | PASS | gqa2/s128/a=346112 ns; gqa2/s4/a=242688 ns; mha4/s128/a=697344 ns; mha4/s4/a=488448 ns; gqa2/s128/b=349184 ns; gqa2/s4/b=242688 ns; mha4/s128/b=695296 ns; mha4/s4/b=491520 ns; gqa2/s128/d=351232 ns; gqa2/s4/d=250880 ns; mha4/s128/d=710656 ns; mha4/s4/d=507904 ns; gqa2/s128/w=342016 ns; gqa2/s4/w=244736 ns; mha4/s128/w=708608 ns; mha4/s4/w=493568 ns |
| A-window | report | PASS | 12 window floor/split checks; gqa2/s128/w1: cp=324608 hol=22872064; gqa2/s4/w1: cp=232448 hol=483328; mha4/s128/w1: cp=685056 hol=47318016; mha4/s4/w1: cp=463872 hol=1936384; gqa2/s128/w2: cp=342016 hol=14131200; gqa2/s4/w2: cp=246784 hol=314368; mha4/s128/w2: cp=684032 hol=44335104; mha4/s4/w2: cp=489472 hol=1215488; gqa2/s128/w4: cp=342016 hol=2158592; gqa2/s4/w4: cp=242688 hol=62464; mha4/s128/w4: cp=688128 hol=23089152; mha4/s4/w4: cp=486400 hol=185344 |
| C1-litmus | hard | PASS | cache/g64/t1024/per_writer=50/50; cache/g64/t1024/thread0_fence=50/50; cache/g64/t1024/no_fence=50/50; cache/g64/t4096/per_writer=50/50; cache/g64/t4096/thread0_fence=50/50; cache/g64/t4096/no_fence=50/50; cache/g128/t1024/per_writer=50/50; cache/g128/t1024/thread0_fence=50/50; cache/g128/t1024/no_fence=50/50; cache/g128/t4096/per_writer=50/50; cache/g128/t4096/thread0_fence=50/50; cache/g128/t4096/no_fence=50/50; cache/g256/t1024/per_writer=50/50; cache/g256/t1024/thread0_fence=50/50; cache/g256/t1024/no_fence=50/50; cache/g256/t4096/per_writer=50/50; cache/g256/t4096/thread0_fence=50/50; cache/g256/t4096/no_fence=50/50; skew/g64/t1024/per_writer=50/50; skew/g64/t1024/thread0_fence=50/50; skew/g64/t1024/no_barrier=50/50; skew/g64/t4096/per_writer=50/50; skew/g64/t4096/thread0_fence=50/50; skew/g64/t4096/no_barrier=50/50; skew/g128/t1024/per_writer=50/50; skew/g128/t1024/thread0_fence=50/50; skew/g128/t1024/no_barrier=50/50; skew/g128/t4096/per_writer=50/50; skew/g128/t4096/thread0_fence=50/50; skew/g128/t4096/no_barrier=50/50; skew/g256/t1024/per_writer=50/50; skew/g256/t1024/thread0_fence=50/50; skew/g256/t1024/no_barrier=50/50; skew/g256/t4096/per_writer=50/50; skew/g256/t4096/thread0_fence=50/50; skew/g256/t4096/no_barrier=50/50 |
| B-a | hard | PASS | gqa2/s4/p0/full=25/25; gqa2/s4/p0/nofence=25/25; gqa2/s4/p0/nowait=25/25; gqa2/s4/p0/neither=25/25; gqa2/s4/p0/l1nosync=25/25; gqa2/s4/p5/full=25/25; gqa2/s4/p5/nofence=25/25; gqa2/s4/p5/nowait=25/25; gqa2/s4/p5/neither=25/25; gqa2/s4/p5/l1nosync=25/25; gqa2/s128/p0/full=25/25; gqa2/s128/p0/nofence=25/25; gqa2/s128/p0/nowait=25/25; gqa2/s128/p0/neither=25/25; gqa2/s128/p0/l1nosync=25/25; gqa2/s128/p5/full=25/25; gqa2/s128/p5/nofence=25/25; gqa2/s128/p5/nowait=25/25; gqa2/s128/p5/neither=25/25; gqa2/s128/p5/l1nosync=25/25; mha4/s4/p0/full=25/25; mha4/s4/p0/nofence=25/25; mha4/s4/p0/nowait=25/25; mha4/s4/p0/neither=25/25; mha4/s4/p0/l1nosync=25/25; mha4/s4/p5/full=25/25; mha4/s4/p5/nofence=25/25; mha4/s4/p5/nowait=25/25; mha4/s4/p5/neither=25/25; mha4/s4/p5/l1nosync=25/25; mha4/s128/p0/full=25/25; mha4/s128/p0/nofence=25/25; mha4/s128/p0/nowait=25/25; mha4/s128/p0/neither=25/25; mha4/s128/p0/l1nosync=25/25; mha4/s128/p5/full=25/25; mha4/s128/p5/nofence=25/25; mha4/s128/p5/nowait=25/25; mha4/s128/p5/neither=25/25; mha4/s128/p5/l1nosync=25/25 |
| B-c/H4-resumed | hard | PASS | Resumed prerequisite order A -> frozen targets -> completed B -> C1 is valid. Original 66ae000e violated H4 and was reverted at 8299ccaa; historical violation is not erased. Correction follows the user-authorized local repair rule. |
| C1-correctness | hard | PASS | gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50 |
| C1-SEQSCAN | hard | PASS | gqa2/s1/p0=50/50; gqa2/s128/p512=50/50; gqa2/s2048/p0=50/50; mha4/s1/p0=50/50; mha4/s128/p512=50/50; mha4/s2048/p0=50/50 |
| C2-correctness | hard | PASS | gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50 |
| C2-SEQSCAN | hard | PASS | gqa2/s1/p0=50/50; gqa2/s128/p512=50/50; gqa2/s2048/p0=50/50; mha4/s1/p0=50/50; mha4/s128/p512=50/50; mha4/s2048/p0=50/50 |
| C2-deadlock | hard | PASS | gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50; gqa2/s4=68 witnesses, gqa2/s128=68 witnesses, mha4/s4=108 witnesses, mha4/s128=140 witnesses |
| C3a-W2 | hard | PASS | gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50 |
| W2-control | hard | PASS | gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50 |
| C3a-W4 | hard | PASS | gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50 |
| W4-control | hard | PASS | gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50 |
| RED-shard-composition | hard | PASS | gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50 |
| C3b-degeneracy | hard | PASS | 2/2 complete sm_89 SASS files identical with cluster-scope switch enabled; freshness checked by H2 |
| C3b-sm120-compile | report | PASS | sm_89 and sm_120 compile; cluster atomic present only at sm_120; sm_120 execution NOT RUN |
| D-a | hard | PASS | gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50 |
| D-b | hard | PASS | gqa2/s4: legacy_grid_stride=17,rotate=19,balanced=17,eft=17,wavefront=19,chain=17; gqa2/s128: legacy_grid_stride=17,rotate=17,balanced=19,eft=17,wavefront=17,chain=17; mha4/s4: legacy_grid_stride=35,rotate=39,balanced=31,eft=35,wavefront=39,chain=36; mha4/s128: legacy_grid_stride=35,rotate=35,balanced=36,eft=36,wavefront=35,chain=35; real/s4: legacy_grid_stride=35,rotate=39,balanced=31,eft=35,wavefront=39,chain=37; real/s128: legacy_grid_stride=31,rotate=35,balanced=35,eft=35,wavefront=35,chain=35 |
| D-c | report | PASS | 6 cells x 4 arms x 25 paired fresh processes; all PASS; gqa2/s4/rotate: L2=0.290816 ms, ratio=1.00000 [1.00000,1.00000]; gqa2/s4/chain: L2=0.291616 ms, ratio=1.00011 [1.00000,1.00330]; gqa2/s4/chain_control: L2=0.292640 ms, ratio=1.00363 [1.00296,1.00671]; gqa2/s4/original: L2=0.292864 ms, ratio=1.00671 [1.00406,1.00749]; gqa2/s128/rotate: L2=0.455680 ms, ratio=1.00000 [1.00000,1.00000]; gqa2/s128/chain: L2=0.500896 ms, ratio=1.09930 [1.09888,1.10112]; gqa2/s128/chain_control: L2=0.514912 ms, ratio=1.12986 [1.12817,1.13034]; gqa2/s128/original: L2=0.518144 ms, ratio=1.13708 [1.13687,1.13708]; mha4/s4/rotate: L2=0.576512 ms, ratio=1.00000 [1.00000,1.00000]; mha4/s4/chain: L2=0.575488 ms, ratio=0.99972 [0.99768,1.00813]; mha4/s4/chain_control: L2=0.581632 ms, ratio=1.01377 [1.01101,1.01593]; mha4/s4/original: L2=0.549856 ms, ratio=1.01223 [1.01040,1.01399]; mha4/s128/rotate: L2=0.855040 ms, ratio=1.00000 [1.00000,1.00000]; mha4/s128/chain: L2=0.967680 ms, ratio=1.13119 [1.13043,1.13291]; mha4/s128/chain_control: L2=1.077248 ms, ratio=1.25988 [1.25854,1.26127]; mha4/s128/original: L2=1.076224 ms, ratio=1.25796 [1.25625,1.26005]; real/s4/rotate: L2=4.407232 ms, ratio=1.00000 [1.00000,1.00000]; real/s4/chain: L2=5.107776 ms, ratio=1.15996 [1.15881,1.16875]; real/s4/chain_control: L2=4.500480 ms, ratio=1.02139 [1.02045,1.02367]; real/s4/original: L2=4.500480 ms, ratio=1.02115 [1.02053,1.02325]; real/s128/rotate: L2=6.499328 ms, ratio=1.00000 [1.00000,1.00000]; real/s128/chain: L2=7.163776 ms, ratio=1.10213 [1.10194,1.10242]; real/s128/chain_control: L2=7.364608 ms, ratio=1.13293 [1.13263,1.13345]; real/s128/original: L2=7.363584 ms, ratio=1.13280 [1.13235,1.13315] |
| D-d | report | PASS | gqa2/s4/cost_model=412; gqa2/s4/trace=1084; mha4/s4/cost_model=908; mha4/s4/trace=3276; real/s4/cost_model=25952; gqa2/s128/cost_model=538124; gqa2/s128/trace=542220; mha4/s128/cost_model=2127388; mha4/s128/trace=2135580; real/s128/cost_model=34394336 |
| E-frozen-targets | report | PASS | 24/24 candidate-specific targets reproduce; 0 targets below their own floor |
| E-target-positions | report | PASS | 24 fixed candidate/cells measured: 16 additional x 3 W=1 configurations x 25; legacy/rotate from protocol matrix |
| window-probe-coverage | hard | PASS | 2/2 safe W=2 SASS images unchanged; nowait BAR.RED sites 1 -> 0; event polling ATOMG 1 -> 0; original partial matrix excluded and retained |
| C-five-arm/ablation | report | PASS | 7 configurations x 8 cells x 5 arms x 25 paired fresh processes |
| real-width-ablation | report | PASS | 5 configurations x 5 arms x 25 fresh processes; all full PASS |
| C1-MEMBAR | report | PASS | gqa2: static L2 MEMBAR.SC.GPU 2 -> 2; publishing release executes on 128 -> 1 threads/task; mha4: static L2 MEMBAR.SC.GPU 2 -> 2; publishing release executes on 128 -> 1 threads/task; prompt static-count expectation corrected: predication changes dynamic participation, not static instruction sites |
| C-instruction-counts | report | PASS | baseline/gqa2/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; baseline/gqa2/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; baseline/mha4/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; baseline/mha4/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; c1/gqa2/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; c1/gqa2/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; c1/mha4/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; c1/mha4/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; c2/gqa2/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; c2/gqa2/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; c2/mha4/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; c2/mha4/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; window2/gqa2/p0: MEMBAR.SC.GPU=3, BAR.SYNC=8; window2/gqa2/p5: MEMBAR.SC.GPU=3, BAR.SYNC=8; window2/mha4/p0: MEMBAR.SC.GPU=3, BAR.SYNC=8; window2/mha4/p5: MEMBAR.SC.GPU=3, BAR.SYNC=8; local2/gqa2/p0: MEMBAR.SC.GPU=3, BAR.SYNC=10; local2/gqa2/p5: MEMBAR.SC.GPU=3, BAR.SYNC=10; local2/mha4/p0: MEMBAR.SC.GPU=3, BAR.SYNC=10; local2/mha4/p5: MEMBAR.SC.GPU=3, BAR.SYNC=10; window4/gqa2/p0: MEMBAR.SC.GPU=3, BAR.SYNC=8; window4/gqa2/p5: MEMBAR.SC.GPU=3, BAR.SYNC=8; window4/mha4/p0: MEMBAR.SC.GPU=3, BAR.SYNC=8; window4/mha4/p5: MEMBAR.SC.GPU=3, BAR.SYNC=8; local4/gqa2/p0: MEMBAR.SC.GPU=3, BAR.SYNC=10; local4/gqa2/p5: MEMBAR.SC.GPU=3, BAR.SYNC=10; local4/mha4/p0: MEMBAR.SC.GPU=3, BAR.SYNC=10; local4/mha4/p5: MEMBAR.SC.GPU=3, BAR.SYNC=10 |
| C3a-neither-barriers | report | PASS | window2/gqa2/neither: BAR.SYNC=6; window2/mha4/neither: BAR.SYNC=6; local2/gqa2/neither: BAR.SYNC=8; local2/mha4/neither: BAR.SYNC=8; local completion convergence remains in neither; full/neither changes reported separately |
| B-b | report | PASS | gqa2/s4/p0: fence=15.360 us, fence/notify=0.4191, fence/protocol=0.2146; gqa2/s4/p5: fence=13.376 us, fence/notify=6.4098, fence/protocol=0.0597; gqa2/s128/p0: fence=22.528 us, fence/notify=0.6642, fence/protocol=0.2857; gqa2/s128/p5: fence=16.480 us, fence/notify=1.0711, fence/protocol=0.0575; mha4/s4/p0: fence=31.744 us, fence/notify=0.4672, fence/protocol=0.2316; mha4/s4/p5: fence=24.576 us, fence/notify=3.5392, fence/protocol=0.0576; mha4/s128/p0: fence=56.160 us, fence/notify=0.7105, fence/protocol=0.3307; mha4/s128/p5: fence=45.056 us, fence/notify=1.0992, fence/protocol=0.0730 |
| C3a-window-gain | report | PASS | gqa2_s4_p0/W2: smem/control=0.96364 [0.96347,0.96568], smem/C2(W1)=1.01562 [1.01199,1.01687]; gqa2_s4_p0/W4: smem/control=0.96373 [0.96136,0.96593], smem/C2(W1)=1.02102 [1.01906,1.02190]; gqa2_s4_p5/W2: smem/control=0.95449 [0.95139,0.95486], smem/C2(W1)=0.98208 [0.98170,0.98466]; gqa2_s4_p5/W4: smem/control=0.95444 [0.95163,0.95486], smem/C2(W1)=0.98214 [0.98195,0.98566]; gqa2_s128_p0/W2: smem/control=0.97577 [0.97433,0.97581], smem/C2(W1)=1.03419 [1.03359,1.03424]; gqa2_s128_p0/W4: smem/control=0.97428 [0.97393,0.97585], smem/C2(W1)=1.03590 [1.03419,1.03628]; gqa2_s128_p5/W2: smem/control=0.97807 [0.97780,0.97992], smem/C2(W1)=1.02507 [1.02480,1.02588]; gqa2_s128_p5/W4: smem/control=0.97817 [0.97598,0.97821], smem/C2(W1)=1.03189 [1.02989,1.03226]; mha4_s4_p0/W2: smem/control=0.96453 [0.96233,0.96577], smem/C2(W1)=1.01166 [1.00957,1.01304]; mha4_s4_p0/W4: smem/control=0.96343 [0.96130,0.96465], smem/C2(W1)=1.01424 [1.01202,1.01452]; mha4_s4_p5/W2: smem/control=0.95288 [0.95280,0.95455], smem/C2(W1)=0.99801 [0.99635,0.99818]; mha4_s4_p5/W4: smem/control=0.95288 [0.95132,0.95459], smem/C2(W1)=0.99971 [0.99636,1.00143]; mha4_s128_p0/W2: smem/control=0.97488 [0.97261,0.97951], smem/C2(W1)=1.04291 [1.02774,1.05428]; mha4_s128_p0/W4: smem/control=0.97715 [0.97418,0.98341], smem/C2(W1)=1.04626 [1.04203,1.05428]; mha4_s128_p5/W2: smem/control=0.97864 [0.95036,1.00335], smem/C2(W1)=1.03815 [1.00868,1.05963]; mha4_s128_p5/W4: smem/control=0.97362 [0.95005,1.02558], smem/C2(W1)=1.03443 [1.00112,1.06433]; all window/shared full logs retain grid=256, 2 CTA/SM and TaskSmem=24576 B; registers 218 -> 220, static shared 0 -> 16 B |
| sm120-self-check | report | PASS | 3/3 runners SELF_CHECK=1 on sm_89; sm_120 has not been run |
| wait+notify<=barrier | research | FAIL | local2: 0/4 median gates; gqa2/s4=1.3625 [1.3136,1.3732], historical excess gap closed=70.53%; gqa2/s128=1.8065 [1.7865,1.8332], historical excess gap closed=45.88%; mha4/s4=1.2784 [1.1873,1.3463], historical excess gap closed=74.69%; mha4/s128=1.7961 [1.5702,2.3186], historical excess gap closed=58.32% |

证据路径随以下完整输出逐门给出。研究门失败不会被转换为硬门通过；sm_120 的执行结果仍为未运行。最终封板后的完整输出位于 [sass_identity/verification.log](sass_identity/verification.log)。

```text
H2	hard	PASS	gqa2=6802641 bytes; mha4=6802641 bytes; input digests match; tested HEAD or artifact-only direct child=True
  evidence: /root/TileMega/docs/experiments/SYNC_V3/sass_identity
A-a	hard	PASS	32/32 floors <= measured
  evidence: /root/TileMega/docs/experiments/PLACE_EFT2/raw/final
A-b	hard	PASS	32/32 split closes, nonnegative, wait <= span
  evidence: /root/TileMega/docs/experiments/PLACE_EFT2/raw/final
A-c	hard	PASS	chain: 20 nodes 242688 ns; rotate: 20 nodes 242688 ns
  evidence: /root/TileMega/docs/experiments/PLACE_EFT2/raw/final
A-d	report	PASS	gqa2/s128/a=346112 ns; gqa2/s4/a=242688 ns; mha4/s128/a=697344 ns; mha4/s4/a=488448 ns; gqa2/s128/b=349184 ns; gqa2/s4/b=242688 ns; mha4/s128/b=695296 ns; mha4/s4/b=491520 ns; gqa2/s128/d=351232 ns; gqa2/s4/d=250880 ns; mha4/s128/d=710656 ns; mha4/s4/d=507904 ns; gqa2/s128/w=342016 ns; gqa2/s4/w=244736 ns; mha4/s128/w=708608 ns; mha4/s4/w=493568 ns
  evidence: /root/TileMega/docs/experiments/PLACE_EFT2/raw/final
A-window	report	PASS	12 window floor/split checks; gqa2/s128/w1: cp=324608 hol=22872064; gqa2/s4/w1: cp=232448 hol=483328; mha4/s128/w1: cp=685056 hol=47318016; mha4/s4/w1: cp=463872 hol=1936384; gqa2/s128/w2: cp=342016 hol=14131200; gqa2/s4/w2: cp=246784 hol=314368; mha4/s128/w2: cp=684032 hol=44335104; mha4/s4/w2: cp=489472 hol=1215488; gqa2/s128/w4: cp=342016 hol=2158592; gqa2/s4/w4: cp=242688 hol=62464; mha4/s128/w4: cp=688128 hol=23089152; mha4/s4/w4: cp=486400 hol=185344
  evidence: /root/TileMega/docs/experiments/WINDOW/raw/final
C1-litmus	hard	PASS	cache/g64/t1024/per_writer=50/50; cache/g64/t1024/thread0_fence=50/50; cache/g64/t1024/no_fence=50/50; cache/g64/t4096/per_writer=50/50; cache/g64/t4096/thread0_fence=50/50; cache/g64/t4096/no_fence=50/50; cache/g128/t1024/per_writer=50/50; cache/g128/t1024/thread0_fence=50/50; cache/g128/t1024/no_fence=50/50; cache/g128/t4096/per_writer=50/50; cache/g128/t4096/thread0_fence=50/50; cache/g128/t4096/no_fence=50/50; cache/g256/t1024/per_writer=50/50; cache/g256/t1024/thread0_fence=50/50; cache/g256/t1024/no_fence=50/50; cache/g256/t4096/per_writer=50/50; cache/g256/t4096/thread0_fence=50/50; cache/g256/t4096/no_fence=50/50; skew/g64/t1024/per_writer=50/50; skew/g64/t1024/thread0_fence=50/50; skew/g64/t1024/no_barrier=50/50; skew/g64/t4096/per_writer=50/50; skew/g64/t4096/thread0_fence=50/50; skew/g64/t4096/no_barrier=50/50; skew/g128/t1024/per_writer=50/50; skew/g128/t1024/thread0_fence=50/50; skew/g128/t1024/no_barrier=50/50; skew/g128/t4096/per_writer=50/50; skew/g128/t4096/thread0_fence=50/50; skew/g128/t4096/no_barrier=50/50; skew/g256/t1024/per_writer=50/50; skew/g256/t1024/thread0_fence=50/50; skew/g256/t1024/no_barrier=50/50; skew/g256/t4096/per_writer=50/50; skew/g256/t4096/thread0_fence=50/50; skew/g256/t4096/no_barrier=50/50
  evidence: /root/TileMega/docs/experiments/SYNC_V3/litmus_v3/scan
B-a	hard	PASS	gqa2/s4/p0/full=25/25; gqa2/s4/p0/nofence=25/25; gqa2/s4/p0/nowait=25/25; gqa2/s4/p0/neither=25/25; gqa2/s4/p0/l1nosync=25/25; gqa2/s4/p5/full=25/25; gqa2/s4/p5/nofence=25/25; gqa2/s4/p5/nowait=25/25; gqa2/s4/p5/neither=25/25; gqa2/s4/p5/l1nosync=25/25; gqa2/s128/p0/full=25/25; gqa2/s128/p0/nofence=25/25; gqa2/s128/p0/nowait=25/25; gqa2/s128/p0/neither=25/25; gqa2/s128/p0/l1nosync=25/25; gqa2/s128/p5/full=25/25; gqa2/s128/p5/nofence=25/25; gqa2/s128/p5/nowait=25/25; gqa2/s128/p5/neither=25/25; gqa2/s128/p5/l1nosync=25/25; mha4/s4/p0/full=25/25; mha4/s4/p0/nofence=25/25; mha4/s4/p0/nowait=25/25; mha4/s4/p0/neither=25/25; mha4/s4/p0/l1nosync=25/25; mha4/s4/p5/full=25/25; mha4/s4/p5/nofence=25/25; mha4/s4/p5/nowait=25/25; mha4/s4/p5/neither=25/25; mha4/s4/p5/l1nosync=25/25; mha4/s128/p0/full=25/25; mha4/s128/p0/nofence=25/25; mha4/s128/p0/nowait=25/25; mha4/s128/p0/neither=25/25; mha4/s128/p0/l1nosync=25/25; mha4/s128/p5/full=25/25; mha4/s128/p5/nofence=25/25; mha4/s128/p5/nowait=25/25; mha4/s128/p5/neither=25/25; mha4/s128/p5/l1nosync=25/25
  evidence: /root/TileMega/docs/experiments/FENCE/raw/paired
B-c/H4-resumed	hard	PASS	Resumed prerequisite order A -> frozen targets -> completed B -> C1 is valid. Original 66ae000e violated H4 and was reverted at 8299ccaa; historical violation is not erased. Correction follows the user-authorized local repair rule.
  evidence: git: 23c5d618 -> 03053089 -> 204df980 -> e6dd44bc
C1-correctness	hard	PASS	gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50
  evidence: /root/TileMega/docs/experiments/SYNC_V3/c1/correctness
C1-SEQSCAN	hard	PASS	gqa2/s1/p0=50/50; gqa2/s128/p512=50/50; gqa2/s2048/p0=50/50; mha4/s1/p0=50/50; mha4/s128/p512=50/50; mha4/s2048/p0=50/50
  evidence: /root/TileMega/docs/experiments/SYNC_V3/c1/seqscan
C2-correctness	hard	PASS	gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50
  evidence: /root/TileMega/docs/experiments/SYNC_V3/c2/correctness
C2-SEQSCAN	hard	PASS	gqa2/s1/p0=50/50; gqa2/s128/p512=50/50; gqa2/s2048/p0=50/50; mha4/s1/p0=50/50; mha4/s128/p512=50/50; mha4/s2048/p0=50/50
  evidence: /root/TileMega/docs/experiments/SYNC_V3/c2/seqscan
C2-deadlock	hard	PASS	gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50; gqa2/s4=68 witnesses, gqa2/s128=68 witnesses, mha4/s4=108 witnesses, mha4/s128=140 witnesses
  evidence: /root/TileMega/docs/experiments/SYNC_V3/c2_dependency
C3a-W2	hard	PASS	gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50
  evidence: /root/TileMega/docs/experiments/SYNC_V3/local2/correctness
W2-control	hard	PASS	gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50
  evidence: /root/TileMega/docs/experiments/SYNC_V3/window2/correctness
C3a-W4	hard	PASS	gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50
  evidence: /root/TileMega/docs/experiments/SYNC_V3/local4/correctness
W4-control	hard	PASS	gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50
  evidence: /root/TileMega/docs/experiments/SYNC_V3/window4/correctness
RED-shard-composition	hard	PASS	gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50
  evidence: /root/TileMega/docs/experiments/SYNC_V3/sharded_red/correctness
C3b-degeneracy	hard	PASS	2/2 complete sm_89 SASS files identical with cluster-scope switch enabled; freshness checked by H2
  evidence: /root/TileMega/docs/experiments/SYNC_V3/sass_identity
C3b-sm120-compile	report	PASS	sm_89 and sm_120 compile; cluster atomic present only at sm_120; sm_120 execution NOT RUN
  evidence: /root/TileMega/docs/experiments/SYNC_V3/cluster_compile
D-a	hard	PASS	gqa2/s4/p3=50/50; gqa2/s128/p3=50/50; mha4/s4/p3=50/50; mha4/s128/p3=50/50
  evidence: /root/TileMega/docs/experiments/CHAIN2/final/correctness
D-b	hard	PASS	gqa2/s4: legacy_grid_stride=17,rotate=19,balanced=17,eft=17,wavefront=19,chain=17; gqa2/s128: legacy_grid_stride=17,rotate=17,balanced=19,eft=17,wavefront=17,chain=17; mha4/s4: legacy_grid_stride=35,rotate=39,balanced=31,eft=35,wavefront=39,chain=36; mha4/s128: legacy_grid_stride=35,rotate=35,balanced=36,eft=36,wavefront=35,chain=35; real/s4: legacy_grid_stride=35,rotate=39,balanced=31,eft=35,wavefront=39,chain=37; real/s128: legacy_grid_stride=31,rotate=35,balanced=35,eft=35,wavefront=35,chain=35
  evidence: /root/TileMega/docs/experiments/CHAIN2/final/replay/path
D-c	report	PASS	6 cells x 4 arms x 25 paired fresh processes; all PASS; gqa2/s4/rotate: L2=0.290816 ms, ratio=1.00000 [1.00000,1.00000]; gqa2/s4/chain: L2=0.291616 ms, ratio=1.00011 [1.00000,1.00330]; gqa2/s4/chain_control: L2=0.292640 ms, ratio=1.00363 [1.00296,1.00671]; gqa2/s4/original: L2=0.292864 ms, ratio=1.00671 [1.00406,1.00749]; gqa2/s128/rotate: L2=0.455680 ms, ratio=1.00000 [1.00000,1.00000]; gqa2/s128/chain: L2=0.500896 ms, ratio=1.09930 [1.09888,1.10112]; gqa2/s128/chain_control: L2=0.514912 ms, ratio=1.12986 [1.12817,1.13034]; gqa2/s128/original: L2=0.518144 ms, ratio=1.13708 [1.13687,1.13708]; mha4/s4/rotate: L2=0.576512 ms, ratio=1.00000 [1.00000,1.00000]; mha4/s4/chain: L2=0.575488 ms, ratio=0.99972 [0.99768,1.00813]; mha4/s4/chain_control: L2=0.581632 ms, ratio=1.01377 [1.01101,1.01593]; mha4/s4/original: L2=0.549856 ms, ratio=1.01223 [1.01040,1.01399]; mha4/s128/rotate: L2=0.855040 ms, ratio=1.00000 [1.00000,1.00000]; mha4/s128/chain: L2=0.967680 ms, ratio=1.13119 [1.13043,1.13291]; mha4/s128/chain_control: L2=1.077248 ms, ratio=1.25988 [1.25854,1.26127]; mha4/s128/original: L2=1.076224 ms, ratio=1.25796 [1.25625,1.26005]; real/s4/rotate: L2=4.407232 ms, ratio=1.00000 [1.00000,1.00000]; real/s4/chain: L2=5.107776 ms, ratio=1.15996 [1.15881,1.16875]; real/s4/chain_control: L2=4.500480 ms, ratio=1.02139 [1.02045,1.02367]; real/s4/original: L2=4.500480 ms, ratio=1.02115 [1.02053,1.02325]; real/s128/rotate: L2=6.499328 ms, ratio=1.00000 [1.00000,1.00000]; real/s128/chain: L2=7.163776 ms, ratio=1.10213 [1.10194,1.10242]; real/s128/chain_control: L2=7.364608 ms, ratio=1.13293 [1.13263,1.13345]; real/s128/original: L2=7.363584 ms, ratio=1.13280 [1.13235,1.13315]
  evidence: /root/TileMega/docs/experiments/CHAIN2/final/paired
D-d	report	PASS	gqa2/s4/cost_model=412; gqa2/s4/trace=1084; mha4/s4/cost_model=908; mha4/s4/trace=3276; real/s4/cost_model=25952; gqa2/s128/cost_model=538124; gqa2/s128/trace=542220; mha4/s128/cost_model=2127388; mha4/s128/trace=2135580; real/s128/cost_model=34394336
  evidence: /root/TileMega/docs/experiments/CHAIN2/final/on/rejected_extensions.tsv
TARGET legacy_grid_stride/gqa2/s4: floor=0.366592000 measured_A=0.453504000 target=0.410048000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s4_legacy_grid_stride
TARGET balanced/gqa2/s4: floor=0.277504000 measured_A=0.648192000 target=0.462848000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s4_balanced
TARGET rotate/gqa2/s4: floor=0.242688000 measured_A=0.291840000 target=0.267264000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s4_rotate
TARGET eft/gqa2/s4: floor=0.246784000 measured_A=0.299232000 target=0.273008000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s4_eft
TARGET wavefront/gqa2/s4: floor=0.242688000 measured_A=0.301056000 target=0.271872000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s4_wavefront
TARGET chain/gqa2/s4: floor=0.243712000 measured_A=0.300032000 target=0.271872000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s4_chain
TARGET legacy_grid_stride/gqa2/s128: floor=0.531456000 measured_A=0.624640000 target=0.578048000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s128_legacy_grid_stride
TARGET balanced/gqa2/s128: floor=0.335872000 measured_A=1.052672000 target=0.694272000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s128_balanced
TARGET rotate/gqa2/s128: floor=0.349184000 measured_A=0.463808000 target=0.406496000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s128_rotate
TARGET eft/gqa2/s128: floor=0.384000000 measured_A=0.466944000 target=0.425472000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s128_eft
TARGET wavefront/gqa2/s128: floor=0.381952000 measured_A=0.466752000 target=0.424352000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s128_wavefront
TARGET chain/gqa2/s128: floor=0.349184000 measured_A=0.528384000 target=0.438784000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/gqa2_s128_chain
TARGET legacy_grid_stride/mha4/s4: floor=0.745472000 measured_A=0.900960000 target=0.823216000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s4_legacy_grid_stride
TARGET balanced/mha4/s4: floor=0.491520000 measured_A=1.236992000 target=0.864256000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s4_balanced
TARGET rotate/mha4/s4: floor=0.488448000 measured_A=0.580608000 target=0.534528000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s4_rotate
TARGET eft/mha4/s4: floor=0.490496000 measured_A=0.595968000 target=0.543232000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s4_eft
TARGET wavefront/mha4/s4: floor=0.492544000 measured_A=0.598144000 target=0.545344000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s4_wavefront
TARGET chain/mha4/s4: floor=0.495616000 measured_A=0.595008000 target=0.545312000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s4_chain
TARGET legacy_grid_stride/mha4/s128: floor=1.074176000 measured_A=1.276928000 target=1.175552000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s128_legacy_grid_stride
TARGET balanced/mha4/s128: floor=0.885760000 measured_A=4.111264000 target=2.498512000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s128_balanced
TARGET rotate/mha4/s128: floor=0.626688000 measured_A=0.866304000 target=0.746496000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s128_rotate
TARGET eft/mha4/s128: floor=0.720896000 measured_A=0.891904000 target=0.806400000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s128_eft
TARGET wavefront/mha4/s128: floor=0.731136000 measured_A=0.960512000 target=0.845824000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s128_wavefront
TARGET chain/mha4/s128: floor=0.648192000 measured_A=1.129472000 target=0.888832000 ms evidence=docs/experiments/SYNC_V3/targets_raw/dump/mha4_s128_chain
E-frozen-targets	report	PASS	24/24 candidate-specific targets reproduce; 0 targets below their own floor
  evidence: /root/TileMega/docs/experiments/SYNC_V3/targets_raw/dump
TARGET_POSITION legacy_grid_stride/gqa2/s4: configuration=baseline L2=0.422912 target=0.410048 delta=0.012864 ms
TARGET_POSITION balanced/gqa2/s4: configuration=baseline L2=0.584896 target=0.462848 delta=0.122048 ms
TARGET_POSITION rotate/gqa2/s4: configuration=c1 L2=0.283648 target=0.267264 delta=0.016384 ms
TARGET_POSITION eft/gqa2/s4: configuration=baseline L2=0.279552 target=0.273008 delta=0.006544 ms
TARGET_POSITION wavefront/gqa2/s4: configuration=baseline L2=0.281600 target=0.271872 delta=0.009728 ms
TARGET_POSITION chain/gqa2/s4: configuration=baseline L2=0.279200 target=0.271872 delta=0.007328 ms
TARGET_POSITION legacy_grid_stride/gqa2/s128: configuration=baseline L2=0.596896 target=0.578048 delta=0.018848 ms
TARGET_POSITION balanced/gqa2/s128: configuration=baseline L2=0.906240 target=0.694272 delta=0.211968 ms
TARGET_POSITION rotate/gqa2/s128: configuration=baseline L2=0.444416 target=0.406496 delta=0.037920 ms
TARGET_POSITION eft/gqa2/s128: configuration=baseline L2=0.439264 target=0.425472 delta=0.013792 ms
TARGET_POSITION wavefront/gqa2/s128: configuration=baseline L2=0.439296 target=0.424352 delta=0.014944 ms
TARGET_POSITION chain/gqa2/s128: configuration=baseline L2=0.493440 target=0.438784 delta=0.054656 ms
TARGET_POSITION legacy_grid_stride/mha4/s4: configuration=baseline L2=0.844800 target=0.823216 delta=0.021584 ms
TARGET_POSITION balanced/mha4/s4: configuration=baseline L2=1.014816 target=0.864256 delta=0.150560 ms
TARGET_POSITION rotate/mha4/s4: configuration=c2 L2=0.561088 target=0.534528 delta=0.026560 ms
TARGET_POSITION eft/mha4/s4: configuration=baseline L2=0.509888 target=0.543232 delta=-0.033344 ms
TARGET_POSITION wavefront/mha4/s4: configuration=baseline L2=0.512832 target=0.545344 delta=-0.032512 ms
TARGET_POSITION chain/mha4/s4: configuration=baseline L2=0.507968 target=0.545312 delta=-0.037344 ms
TARGET_POSITION legacy_grid_stride/mha4/s128: configuration=baseline L2=1.211392 target=1.175552 delta=0.035840 ms
TARGET_POSITION balanced/mha4/s128: configuration=baseline L2=3.286016 target=2.498512 delta=0.787504 ms
TARGET_POSITION rotate/mha4/s128: configuration=baseline L2=0.870400 target=0.746496 delta=0.123904 ms
TARGET_POSITION eft/mha4/s128: configuration=baseline L2=0.836576 target=0.806400 delta=0.030176 ms
TARGET_POSITION wavefront/mha4/s128: configuration=baseline L2=0.820224 target=0.845824 delta=-0.025600 ms
TARGET_POSITION chain/mha4/s128: configuration=baseline L2=1.033216 target=0.888832 delta=0.144384 ms
E-target-positions	report	PASS	24 fixed candidate/cells measured: 16 additional x 3 W=1 configurations x 25; legacy/rotate from protocol matrix
  evidence: /root/TileMega/docs/experiments/SYNC_V3/target_positions
window-probe-coverage	hard	PASS	2/2 safe W=2 SASS images unchanged; nowait BAR.RED sites 1 -> 0; event polling ATOMG 1 -> 0; original partial matrix excluded and retained
  evidence: /root/TileMega/docs/experiments/SYNC_V3/window_probe_fix
ABLATION baseline/gqa2/s4/p0 l2_ms=0.422912 nofence_ms=0.408576 nowait_ms=0.389120 neither_ms=0.353280 l1nosync_ms=0.364544 fence_ms=0.015168 wait_ms=0.034816 notify_ms=0.034816 barrier_ms=0.026784 protocol_over_barrier=2.576923 l2_over_l1=1.081008 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/baseline/paired/gqa2_s4_p0
ABLATION baseline/gqa2/s4/p5 l2_ms=0.284576 nofence_ms=0.270336 nowait_ms=0.062464 neither_ms=0.059392 l1nosync_ms=0.365568 fence_ms=0.014272 wait_ms=0.222208 notify_ms=0.003072 barrier_ms=0.026688 protocol_over_barrier=8.441247 l2_over_l1=0.723998 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/baseline/paired/gqa2_s4_p5
ABLATION baseline/gqa2/s128/p0 l2_ms=0.596896 nofence_ms=0.574272 nowait_ms=0.551936 neither_ms=0.518944 l1nosync_ms=0.526336 fence_ms=0.022528 wait_ms=0.044032 notify_ms=0.033792 barrier_ms=0.032608 protocol_over_barrier=2.404810 l2_over_l1=1.067766 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/baseline/paired/gqa2_s128_p0
ABLATION baseline/gqa2/s128/p5 l2_ms=0.444416 nofence_ms=0.427808 nowait_ms=0.172192 neither_ms=0.156672 l1nosync_ms=0.526336 fence_ms=0.017248 wait_ms=0.272288 notify_ms=0.015360 barrier_ms=0.031744 protocol_over_barrier=9.062500 l2_over_l1=0.796330 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/baseline/paired/gqa2_s128_p5
ABLATION baseline/mha4/s4/p0 l2_ms=0.844800 nofence_ms=0.816096 nowait_ms=0.779264 neither_ms=0.710656 l1nosync_ms=0.728064 fence_ms=0.029920 wait_ms=0.066560 notify_ms=0.067552 barrier_ms=0.053504 protocol_over_barrier=2.479742 l2_over_l1=1.081258 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/baseline/paired/mha4_s4_p0
ABLATION baseline/mha4/s4/p5 l2_ms=0.562176 nofence_ms=0.537600 nowait_ms=0.142144 neither_ms=0.135168 l1nosync_ms=0.728064 fence_ms=0.024576 wait_ms=0.420096 notify_ms=0.006880 barrier_ms=0.053440 protocol_over_barrier=7.859835 l2_over_l1=0.718984 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/baseline/paired/mha4_s4_p5
ABLATION baseline/mha4/s128/p0 l2_ms=1.211392 nofence_ms=1.139712 nowait_ms=1.065984 neither_ms=0.997088 l1nosync_ms=1.038336 fence_ms=0.055296 wait_ms=0.135040 notify_ms=0.073536 barrier_ms=0.062656 protocol_over_barrier=2.247934 l2_over_l1=1.083257 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/baseline/paired/mha4_s128_p0
ABLATION baseline/mha4/s128/p5 l2_ms=0.870400 nofence_ms=0.861184 nowait_ms=0.328768 neither_ms=0.288768 l1nosync_ms=1.056768 fence_ms=0.045056 wait_ms=0.541408 notify_ms=0.039936 barrier_ms=0.009184 protocol_over_barrier=9.882864 l2_over_l1=0.815143 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/baseline/paired/mha4_s128_p5
ABLATION c1/gqa2/s4/p0 l2_ms=0.427008 nofence_ms=0.408576 nowait_ms=0.390144 neither_ms=0.354176 l1nosync_ms=0.364544 fence_ms=0.018112 wait_ms=0.035936 notify_ms=0.036864 barrier_ms=0.026464 protocol_over_barrier=2.784764 l2_over_l1=1.089507 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c1/paired/gqa2_s4_p0
ABLATION c1/gqa2/s4/p5 l2_ms=0.283648 nofence_ms=0.270272 nowait_ms=0.061536 neither_ms=0.059360 l1nosync_ms=0.365568 fence_ms=0.013472 wait_ms=0.221568 notify_ms=0.003072 barrier_ms=0.026784 protocol_over_barrier=8.384615 l2_over_l1=0.723238 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c1/paired/gqa2_s4_p5
ABLATION c1/gqa2/s128/p0 l2_ms=0.599008 nofence_ms=0.573504 nowait_ms=0.553984 neither_ms=0.519072 l1nosync_ms=0.526336 fence_ms=0.024576 wait_ms=0.045056 notify_ms=0.034848 barrier_ms=0.031744 protocol_over_barrier=2.530531 l2_over_l1=1.072934 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c1/paired/gqa2_s128_p0
ABLATION c1/gqa2/s128/p5 l2_ms=0.444576 nofence_ms=0.428000 nowait_ms=0.173056 neither_ms=0.156672 l1nosync_ms=0.526336 fence_ms=0.017408 wait_ms=0.271552 notify_ms=0.016160 barrier_ms=0.031840 protocol_over_barrier=9.049000 l2_over_l1=0.797473 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c1/paired/gqa2_s128_p5
ABLATION c1/mha4/s4/p0 l2_ms=0.849952 nofence_ms=0.815104 nowait_ms=0.779264 neither_ms=0.710656 l1nosync_ms=0.729088 fence_ms=0.033792 wait_ms=0.069664 notify_ms=0.069632 barrier_ms=0.052416 protocol_over_barrier=2.627451 l2_over_l1=1.087685 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c1/paired/mha4_s4_p0
ABLATION c1/mha4/s4/p5 l2_ms=0.561152 nofence_ms=0.537568 nowait_ms=0.142208 neither_ms=0.135040 l1nosync_ms=0.729024 fence_ms=0.023552 wait_ms=0.418816 notify_ms=0.007072 barrier_ms=0.053184 protocol_over_barrier=8.004207 l2_over_l1=0.717589 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c1/paired/mha4_s4_p5
ABLATION c1/mha4/s128/p0 l2_ms=1.218560 nofence_ms=1.130496 nowait_ms=1.090560 neither_ms=1.014784 l1nosync_ms=1.032032 fence_ms=0.063296 wait_ms=0.107904 notify_ms=0.076800 barrier_ms=0.078016 protocol_over_barrier=2.360456 l2_over_l1=1.089591 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c1/paired/mha4_s128_p0
ABLATION c1/mha4/s128/p5 l2_ms=0.889856 nofence_ms=0.846784 nowait_ms=0.328704 neither_ms=0.288768 l1nosync_ms=1.055744 fence_ms=0.050176 wait_ms=0.576512 notify_ms=0.038912 barrier_ms=0.052416 protocol_over_barrier=9.918033 l2_over_l1=0.812244 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c1/paired/mha4_s128_p5
ABLATION c2/gqa2/s4/p0 l2_ms=0.427008 nofence_ms=0.408544 nowait_ms=0.389120 neither_ms=0.354304 l1nosync_ms=0.364544 fence_ms=0.018176 wait_ms=0.036704 notify_ms=0.035232 barrier_ms=0.026816 protocol_over_barrier=2.705602 l2_over_l1=1.089865 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c2/paired/gqa2_s4_p0
ABLATION c2/gqa2/s4/p5 l2_ms=0.285600 nofence_ms=0.270336 nowait_ms=0.062208 neither_ms=0.059392 l1nosync_ms=0.365472 fence_ms=0.015232 wait_ms=0.223424 notify_ms=0.003040 barrier_ms=0.026720 protocol_over_barrier=8.474251 l2_over_l1=0.727749 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c2/paired/gqa2_s4_p5
ABLATION c2/gqa2/s128/p0 l2_ms=0.599040 nofence_ms=0.573568 nowait_ms=0.553984 neither_ms=0.518336 l1nosync_ms=0.526336 fence_ms=0.025472 wait_ms=0.045056 notify_ms=0.034848 barrier_ms=0.032896 protocol_over_barrier=2.454369 l2_over_l1=1.071952 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c2/paired/gqa2_s128_p0
ABLATION c2/gqa2/s128/p5 l2_ms=0.445312 nofence_ms=0.428032 nowait_ms=0.172960 neither_ms=0.156704 l1nosync_ms=0.526336 fence_ms=0.017216 wait_ms=0.272416 notify_ms=0.015360 barrier_ms=0.031936 protocol_over_barrier=9.006012 l2_over_l1=0.797794 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c2/paired/gqa2_s128_p5
ABLATION c2/mha4/s4/p0 l2_ms=0.851968 nofence_ms=0.815296 nowait_ms=0.782336 neither_ms=0.711680 l1nosync_ms=0.728064 fence_ms=0.036864 wait_ms=0.070656 notify_ms=0.072704 barrier_ms=0.052448 protocol_over_barrier=2.660377 l2_over_l1=1.091623 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c2/paired/mha4_s4_p0
ABLATION c2/mha4/s4/p5 l2_ms=0.561088 nofence_ms=0.537504 nowait_ms=0.142208 neither_ms=0.135168 l1nosync_ms=0.728064 fence_ms=0.023552 wait_ms=0.418816 notify_ms=0.007104 barrier_ms=0.054272 protocol_over_barrier=7.872866 l2_over_l1=0.718136 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c2/paired/mha4_s4_p5
ABLATION c2/mha4/s128/p0 l2_ms=1.219616 nofence_ms=1.156960 nowait_ms=1.097728 neither_ms=1.008640 l1nosync_ms=1.054720 fence_ms=0.063488 wait_ms=0.101376 notify_ms=0.079072 barrier_ms=0.067584 protocol_over_barrier=2.594223 l2_over_l1=1.090494 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c2/paired/mha4_s128_p0
ABLATION c2/mha4/s128/p5 l2_ms=0.884736 nofence_ms=0.844672 nowait_ms=0.326912 neither_ms=0.288768 l1nosync_ms=1.056768 fence_ms=0.050176 wait_ms=0.567552 notify_ms=0.039936 barrier_ms=0.029856 protocol_over_barrier=9.918033 l2_over_l1=0.814306 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/c2/paired/mha4_s128_p5
ABLATION window2/gqa2/s4/p0 l2_ms=0.449536 nofence_ms=0.436064 nowait_ms=0.427008 neither_ms=0.374784 l1nosync_ms=0.364704 fence_ms=0.014336 wait_ms=0.022528 notify_ms=0.052224 barrier_ms=0.027456 protocol_over_barrier=2.764423 l2_over_l1=1.146417 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window2/paired/gqa2_s4_p0
ABLATION window2/gqa2/s4/p5 l2_ms=0.294048 nofence_ms=0.280576 nowait_ms=0.064512 neither_ms=0.061440 l1nosync_ms=0.365568 fence_ms=0.013312 wait_ms=0.229376 notify_ms=0.002944 barrier_ms=0.026752 protocol_over_barrier=8.668258 l2_over_l1=0.749634 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window2/paired/gqa2_s4_p5
ABLATION window2/gqa2/s128/p0 l2_ms=0.634880 nofence_ms=0.615360 nowait_ms=0.594944 neither_ms=0.539648 l1nosync_ms=0.526336 fence_ms=0.020256 wait_ms=0.040000 notify_ms=0.055296 barrier_ms=0.031808 protocol_over_barrier=3.000000 l2_over_l1=1.137876 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window2/paired/gqa2_s128_p0
ABLATION window2/gqa2/s128/p5 l2_ms=0.466944 nofence_ms=0.451584 nowait_ms=0.185344 neither_ms=0.169984 l1nosync_ms=0.526208 fence_ms=0.014560 wait_ms=0.281536 notify_ms=0.016160 barrier_ms=0.031904 protocol_over_barrier=9.322581 l2_over_l1=0.836409 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window2/paired/gqa2_s128_p5
ABLATION window2/mha4/s4/p0 l2_ms=0.893824 nofence_ms=0.866240 nowait_ms=0.849920 neither_ms=0.749568 l1nosync_ms=0.728064 fence_ms=0.027648 wait_ms=0.044000 notify_ms=0.101376 barrier_ms=0.053248 protocol_over_barrier=2.705882 l2_over_l1=1.144004 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window2/paired/mha4_s4_p0
ABLATION window2/mha4/s4/p5 l2_ms=0.587776 nofence_ms=0.561984 nowait_ms=0.147456 neither_ms=0.147456 l1nosync_ms=0.729088 fence_ms=0.024768 wait_ms=0.440320 notify_ms=0.000000 barrier_ms=0.053696 protocol_over_barrier=8.051613 l2_over_l1=0.749494 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window2/paired/mha4_s4_p5
ABLATION window2/mha4/s128/p0 l2_ms=1.304576 nofence_ms=1.222656 nowait_ms=1.171456 neither_ms=1.035264 l1nosync_ms=1.055904 fence_ms=0.058368 wait_ms=0.130048 notify_ms=0.117760 barrier_ms=0.065408 protocol_over_barrier=3.267267 l2_over_l1=1.165064 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window2/paired/mha4_s128_p0
ABLATION window2/mha4/s128/p5 l2_ms=0.956416 nofence_ms=0.905152 nowait_ms=0.361472 neither_ms=0.325632 l1nosync_ms=1.057792 fence_ms=0.038912 wait_ms=0.591872 notify_ms=0.035840 barrier_ms=0.057344 protocol_over_barrier=10.403917 l2_over_l1=0.859361 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window2/paired/mha4_s128_p5
ABLATION local2/gqa2/s4/p0 l2_ms=0.433152 nofence_ms=0.416768 nowait_ms=0.414688 neither_ms=0.397312 l1nosync_ms=0.364416 fence_ms=0.017280 wait_ms=0.019296 notify_ms=0.017664 barrier_ms=0.027552 protocol_over_barrier=1.362530 l2_over_l1=1.104706 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local2/paired/gqa2_s4_p0
ABLATION local2/gqa2/s4/p5 l2_ms=0.280576 nofence_ms=0.270336 nowait_ms=0.063488 neither_ms=0.063232 l1nosync_ms=0.364544 fence_ms=0.010240 wait_ms=0.217088 notify_ms=0.000160 barrier_ms=0.027456 protocol_over_barrier=7.944056 l2_over_l1=0.714996 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local2/paired/gqa2_s4_p5
ABLATION local2/gqa2/s128/p0 l2_ms=0.619520 nofence_ms=0.596992 nowait_ms=0.579584 neither_ms=0.561152 l1nosync_ms=0.525408 fence_ms=0.022528 wait_ms=0.039936 notify_ms=0.018496 barrier_ms=0.031968 protocol_over_barrier=1.806452 l2_over_l1=1.110092 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local2/paired/gqa2_s128_p0
ABLATION local2/gqa2/s128/p5 l2_ms=0.456576 nofence_ms=0.441344 nowait_ms=0.185600 neither_ms=0.173824 l1nosync_ms=0.526336 fence_ms=0.014464 wait_ms=0.270432 notify_ms=0.012128 barrier_ms=0.031744 protocol_over_barrier=8.903226 l2_over_l1=0.818177 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local2/paired/gqa2_s128_p5
ABLATION local2/mha4/s4/p0 l2_ms=0.862208 nofence_ms=0.830464 nowait_ms=0.825344 neither_ms=0.792576 l1nosync_ms=0.728064 fence_ms=0.030784 wait_ms=0.036608 notify_ms=0.034656 barrier_ms=0.054272 protocol_over_barrier=1.278399 l2_over_l1=1.102046 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local2/paired/mha4_s4_p0
ABLATION local2/mha4/s4/p5 l2_ms=0.560064 nofence_ms=0.540672 nowait_ms=0.144384 neither_ms=0.142592 l1nosync_ms=0.729088 fence_ms=0.018432 wait_ms=0.415744 notify_ms=0.001088 barrier_ms=0.055296 protocol_over_barrier=7.553819 l2_over_l1=0.714735 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local2/paired/mha4_s4_p5
ABLATION local2/mha4/s128/p0 l2_ms=1.272832 nofence_ms=1.197056 nowait_ms=1.159168 neither_ms=1.104896 l1nosync_ms=1.034240 fence_ms=0.062432 wait_ms=0.103648 notify_ms=0.048128 barrier_ms=0.078848 protocol_over_barrier=1.796076 l2_over_l1=1.135160 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local2/paired/mha4_s128_p0
ABLATION local2/mha4/s128/p5 l2_ms=0.919552 nofence_ms=0.878592 nowait_ms=0.360704 neither_ms=0.335648 l1nosync_ms=1.057792 fence_ms=0.042720 wait_ms=0.570368 notify_ms=0.025056 barrier_ms=0.044960 protocol_over_barrier=9.737705 l2_over_l1=0.842922 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local2/paired/mha4_s128_p5
ABLATION window4/gqa2/s4/p0 l2_ms=0.451584 nofence_ms=0.438272 nowait_ms=0.427872 neither_ms=0.374784 l1nosync_ms=0.364544 fence_ms=0.013184 wait_ms=0.024672 notify_ms=0.052896 barrier_ms=0.027648 protocol_over_barrier=2.790509 l2_over_l1=1.150873 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window4/paired/gqa2_s4_p0
ABLATION window4/gqa2/s4/p5 l2_ms=0.294112 nofence_ms=0.281440 nowait_ms=0.064544 neither_ms=0.062464 l1nosync_ms=0.364544 fence_ms=0.013088 wait_ms=0.229376 notify_ms=0.003040 barrier_ms=0.027616 protocol_over_barrier=8.451912 l2_over_l1=0.750082 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window4/paired/gqa2_s4_p5
ABLATION window4/gqa2/s128/p0 l2_ms=0.636704 nofence_ms=0.616448 nowait_ms=0.594944 neither_ms=0.539648 l1nosync_ms=0.526336 fence_ms=0.020256 wait_ms=0.041984 notify_ms=0.055296 barrier_ms=0.031904 protocol_over_barrier=3.023116 l2_over_l1=1.141007 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window4/paired/gqa2_s128_p0
ABLATION window4/gqa2/s128/p5 l2_ms=0.470016 nofence_ms=0.453632 nowait_ms=0.185472 neither_ms=0.169984 l1nosync_ms=0.525312 fence_ms=0.016384 wait_ms=0.283648 notify_ms=0.015712 barrier_ms=0.031744 protocol_over_barrier=9.419355 l2_over_l1=0.841960 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window4/paired/gqa2_s128_p5
ABLATION window4/mha4/s4/p0 l2_ms=0.897024 nofence_ms=0.871424 nowait_ms=0.850944 neither_ms=0.748448 l1nosync_ms=0.728064 fence_ms=0.026624 wait_ms=0.047040 notify_ms=0.103424 barrier_ms=0.054432 protocol_over_barrier=2.773585 l2_over_l1=1.149010 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window4/paired/mha4_s4_p0
ABLATION window4/mha4/s4/p5 l2_ms=0.587776 nofence_ms=0.562176 nowait_ms=0.147456 neither_ms=0.147456 l1nosync_ms=0.729088 fence_ms=0.025504 wait_ms=0.440256 notify_ms=0.000000 barrier_ms=0.054304 protocol_over_barrier=8.111373 l2_over_l1=0.750143 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window4/paired/mha4_s4_p5
ABLATION window4/mha4/s128/p0 l2_ms=1.302528 nofence_ms=1.218560 nowait_ms=1.173504 neither_ms=1.054720 l1nosync_ms=1.056768 fence_ms=0.053344 wait_ms=0.126976 notify_ms=0.112800 barrier_ms=0.063456 protocol_over_barrier=3.351746 l2_over_l1=1.165246 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window4/paired/mha4_s128_p0
ABLATION window4/mha4/s128/p5 l2_ms=0.923648 nofence_ms=0.880640 nowait_ms=0.360448 neither_ms=0.324608 l1nosync_ms=1.057792 fence_ms=0.044032 wait_ms=0.577536 notify_ms=0.035840 barrier_ms=0.033056 protocol_over_barrier=10.185129 l2_over_l1=0.864611 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/window4/paired/mha4_s128_p5
ABLATION local4/gqa2/s4/p0 l2_ms=0.435200 nofence_ms=0.419840 nowait_ms=0.414720 neither_ms=0.396288 l1nosync_ms=0.365568 fence_ms=0.016160 wait_ms=0.020480 notify_ms=0.018432 barrier_ms=0.026784 protocol_over_barrier=1.462772 l2_over_l1=1.109130 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local4/paired/gqa2_s4_p0
ABLATION local4/gqa2/s4/p5 l2_ms=0.280576 nofence_ms=0.270336 nowait_ms=0.063488 neither_ms=0.062464 l1nosync_ms=0.365568 fence_ms=0.010240 wait_ms=0.217088 notify_ms=0.000320 barrier_ms=0.027456 protocol_over_barrier=7.935897 l2_over_l1=0.715405 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local4/paired/gqa2_s4_p5
ABLATION local4/gqa2/s128/p0 l2_ms=0.620544 nofence_ms=0.599008 nowait_ms=0.579584 neither_ms=0.561152 l1nosync_ms=0.526176 fence_ms=0.021504 wait_ms=0.040960 notify_ms=0.018400 barrier_ms=0.031968 protocol_over_barrier=1.838710 l2_over_l1=1.111788 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local4/paired/gqa2_s128_p0
ABLATION local4/gqa2/s128/p5 l2_ms=0.458752 nofence_ms=0.443392 nowait_ms=0.185344 neither_ms=0.174080 l1nosync_ms=0.525376 fence_ms=0.016096 wait_ms=0.273408 notify_ms=0.011264 barrier_ms=0.031872 protocol_over_barrier=8.944556 l2_over_l1=0.823198 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local4/paired/gqa2_s128_p5
ABLATION local4/mha4/s4/p0 l2_ms=0.864256 nofence_ms=0.834368 nowait_ms=0.826304 neither_ms=0.791552 l1nosync_ms=0.728032 fence_ms=0.030080 wait_ms=0.037152 notify_ms=0.034816 barrier_ms=0.054496 protocol_over_barrier=1.306122 l2_over_l1=1.104578 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local4/paired/mha4_s4_p0
ABLATION local4/mha4/s4/p5 l2_ms=0.560128 nofence_ms=0.540672 nowait_ms=0.144384 neither_ms=0.142336 l1nosync_ms=0.729088 fence_ms=0.019392 wait_ms=0.415776 notify_ms=0.001664 barrier_ms=0.054432 protocol_over_barrier=7.666471 l2_over_l1=0.714858 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local4/paired/mha4_s4_p5
ABLATION local4/mha4/s128/p0 l2_ms=1.274880 nofence_ms=1.193984 nowait_ms=1.143808 neither_ms=1.098592 l1nosync_ms=1.027072 fence_ms=0.057344 wait_ms=0.119936 notify_ms=0.047200 barrier_ms=0.083040 protocol_over_barrier=1.751493 l2_over_l1=1.137705 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local4/paired/mha4_s128_p0
ABLATION local4/mha4/s128/p5 l2_ms=0.914432 nofence_ms=0.866304 nowait_ms=0.361472 neither_ms=0.336640 l1nosync_ms=1.057792 fence_ms=0.041984 wait_ms=0.551968 notify_ms=0.025600 barrier_ms=0.020640 protocol_over_barrier=9.342714 l2_over_l1=0.846910 evidence=/root/TileMega/docs/experiments/SYNC_V3/ablation/local4/paired/mha4_s128_p5
C-five-arm/ablation	report	PASS	7 configurations x 8 cells x 5 arms x 25 paired fresh processes
  evidence: /root/TileMega/docs/experiments/SYNC_V3/ablation
real-width-ablation	report	PASS	5 configurations x 5 arms x 25 fresh processes; all full PASS
  evidence: /root/TileMega/docs/experiments/SYNC_V3/realwidth
C1-MEMBAR	report	PASS	gqa2: static L2 MEMBAR.SC.GPU 2 -> 2; publishing release executes on 128 -> 1 threads/task; mha4: static L2 MEMBAR.SC.GPU 2 -> 2; publishing release executes on 128 -> 1 threads/task; prompt static-count expectation corrected: predication changes dynamic participation, not static instruction sites
  evidence: /root/TileMega/docs/experiments/SYNC_V3/c1/sass
C-instruction-counts	report	PASS	baseline/gqa2/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; baseline/gqa2/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; baseline/mha4/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; baseline/mha4/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; c1/gqa2/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; c1/gqa2/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; c1/mha4/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; c1/mha4/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; c2/gqa2/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; c2/gqa2/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; c2/mha4/p0: MEMBAR.SC.GPU=2, BAR.SYNC=8; c2/mha4/p5: MEMBAR.SC.GPU=2, BAR.SYNC=8; window2/gqa2/p0: MEMBAR.SC.GPU=3, BAR.SYNC=8; window2/gqa2/p5: MEMBAR.SC.GPU=3, BAR.SYNC=8; window2/mha4/p0: MEMBAR.SC.GPU=3, BAR.SYNC=8; window2/mha4/p5: MEMBAR.SC.GPU=3, BAR.SYNC=8; local2/gqa2/p0: MEMBAR.SC.GPU=3, BAR.SYNC=10; local2/gqa2/p5: MEMBAR.SC.GPU=3, BAR.SYNC=10; local2/mha4/p0: MEMBAR.SC.GPU=3, BAR.SYNC=10; local2/mha4/p5: MEMBAR.SC.GPU=3, BAR.SYNC=10; window4/gqa2/p0: MEMBAR.SC.GPU=3, BAR.SYNC=8; window4/gqa2/p5: MEMBAR.SC.GPU=3, BAR.SYNC=8; window4/mha4/p0: MEMBAR.SC.GPU=3, BAR.SYNC=8; window4/mha4/p5: MEMBAR.SC.GPU=3, BAR.SYNC=8; local4/gqa2/p0: MEMBAR.SC.GPU=3, BAR.SYNC=10; local4/gqa2/p5: MEMBAR.SC.GPU=3, BAR.SYNC=10; local4/mha4/p0: MEMBAR.SC.GPU=3, BAR.SYNC=10; local4/mha4/p5: MEMBAR.SC.GPU=3, BAR.SYNC=10
  evidence: /root/TileMega/docs/experiments/SYNC_V3/c2/sass
C3a-neither-barriers	report	PASS	window2/gqa2/neither: BAR.SYNC=6; window2/mha4/neither: BAR.SYNC=6; local2/gqa2/neither: BAR.SYNC=8; local2/mha4/neither: BAR.SYNC=8; local completion convergence remains in neither; full/neither changes reported separately
  evidence: /root/TileMega/docs/experiments/SYNC_V3/local_probe_sass
B-b	report	PASS	gqa2/s4/p0: fence=15.360 us, fence/notify=0.4191, fence/protocol=0.2146; gqa2/s4/p5: fence=13.376 us, fence/notify=6.4098, fence/protocol=0.0597; gqa2/s128/p0: fence=22.528 us, fence/notify=0.6642, fence/protocol=0.2857; gqa2/s128/p5: fence=16.480 us, fence/notify=1.0711, fence/protocol=0.0575; mha4/s4/p0: fence=31.744 us, fence/notify=0.4672, fence/protocol=0.2316; mha4/s4/p5: fence=24.576 us, fence/notify=3.5392, fence/protocol=0.0576; mha4/s128/p0: fence=56.160 us, fence/notify=0.7105, fence/protocol=0.3307; mha4/s128/p5: fence=45.056 us, fence/notify=1.0992, fence/protocol=0.0730
  evidence: /root/TileMega/docs/experiments/FENCE/raw/paired
C3a-window-gain	report	PASS	gqa2_s4_p0/W2: smem/control=0.96364 [0.96347,0.96568], smem/C2(W1)=1.01562 [1.01199,1.01687]; gqa2_s4_p0/W4: smem/control=0.96373 [0.96136,0.96593], smem/C2(W1)=1.02102 [1.01906,1.02190]; gqa2_s4_p5/W2: smem/control=0.95449 [0.95139,0.95486], smem/C2(W1)=0.98208 [0.98170,0.98466]; gqa2_s4_p5/W4: smem/control=0.95444 [0.95163,0.95486], smem/C2(W1)=0.98214 [0.98195,0.98566]; gqa2_s128_p0/W2: smem/control=0.97577 [0.97433,0.97581], smem/C2(W1)=1.03419 [1.03359,1.03424]; gqa2_s128_p0/W4: smem/control=0.97428 [0.97393,0.97585], smem/C2(W1)=1.03590 [1.03419,1.03628]; gqa2_s128_p5/W2: smem/control=0.97807 [0.97780,0.97992], smem/C2(W1)=1.02507 [1.02480,1.02588]; gqa2_s128_p5/W4: smem/control=0.97817 [0.97598,0.97821], smem/C2(W1)=1.03189 [1.02989,1.03226]; mha4_s4_p0/W2: smem/control=0.96453 [0.96233,0.96577], smem/C2(W1)=1.01166 [1.00957,1.01304]; mha4_s4_p0/W4: smem/control=0.96343 [0.96130,0.96465], smem/C2(W1)=1.01424 [1.01202,1.01452]; mha4_s4_p5/W2: smem/control=0.95288 [0.95280,0.95455], smem/C2(W1)=0.99801 [0.99635,0.99818]; mha4_s4_p5/W4: smem/control=0.95288 [0.95132,0.95459], smem/C2(W1)=0.99971 [0.99636,1.00143]; mha4_s128_p0/W2: smem/control=0.97488 [0.97261,0.97951], smem/C2(W1)=1.04291 [1.02774,1.05428]; mha4_s128_p0/W4: smem/control=0.97715 [0.97418,0.98341], smem/C2(W1)=1.04626 [1.04203,1.05428]; mha4_s128_p5/W2: smem/control=0.97864 [0.95036,1.00335], smem/C2(W1)=1.03815 [1.00868,1.05963]; mha4_s128_p5/W4: smem/control=0.97362 [0.95005,1.02558], smem/C2(W1)=1.03443 [1.00112,1.06433]; all window/shared full logs retain grid=256, 2 CTA/SM and TaskSmem=24576 B; registers 218 -> 220, static shared 0 -> 16 B
  evidence: /root/TileMega/docs/experiments/SYNC_V3/ablation
sm120-self-check	report	PASS	3/3 runners SELF_CHECK=1 on sm_89; sm_120 has not been run
  evidence: /root/TileMega/docs/experiments/SYNC_V3/self_check
RESEARCH_CONFIGURATION baseline: achieved=0/4 protocol_ratio_geomean=2.424337 full_L2_geomean_ms=0.712930
RESEARCH_CONFIGURATION c1: achieved=0/4 protocol_ratio_geomean=2.571181 full_L2_geomean_ms=0.717427
RESEARCH_CONFIGURATION c2: achieved=0/4 protocol_ratio_geomean=2.601888 full_L2_geomean_ms=0.718017
RESEARCH_CONFIGURATION window2: achieved=0/4 protocol_ratio_geomean=2.926206 full_L2_geomean_ms=0.759529
RESEARCH_CONFIGURATION local2: achieved=0/4 protocol_ratio_geomean=1.541846 full_L2_geomean_ms=0.736664
RESEARCH_CONFIGURATION window4: achieved=0/4 protocol_ratio_geomean=2.975861 full_L2_geomean_ms=0.761319
RESEARCH_CONFIGURATION local4: achieved=0/4 protocol_ratio_geomean=1.574964 full_L2_geomean_ms=0.738573
wait+notify<=barrier	research	FAIL	local2: 0/4 median gates; gqa2/s4=1.3625 [1.3136,1.3732], historical excess gap closed=70.53%; gqa2/s128=1.8065 [1.7865,1.8332], historical excess gap closed=45.88%; mha4/s4=1.2784 [1.1873,1.3463], historical excess gap closed=74.69%; mha4/s128=1.7961 [1.5702,2.3186], historical excess gap closed=58.32%
  evidence: /root/TileMega/docs/experiments/SYNC_V3/ablation

21/21 hard gates pass; 15/15 report gates pass

```

## 3. 唯一研究门的位置

默认放置始终为 `TILEMEGA_PLACEMENT=0`，每格 25 轮同会话配对、轮转臂序、全新进程。区间为配对比值中位数的 10,000 次 bootstrap 95% CI（seed=167）。七组配置均在完整四格上评估；至少一个配置达成三格才算研究门通过。代表配置优先选择达门格数最多者，再以四格协议/barrier 中位比的几何平均打破平局。最快端到端配置另行记录；窗口仍仅显式开启。差距缩减比例为 `(历史倍数−本轮倍数)/(历史倍数−1)`，允许负值。

| cell | 配置 | 历史倍数 | 本轮倍数 | 95% CI | 缩掉历史差距 | 达门 |
|---|---|---|---|---|---|---|
| gqa2 s4 | local2 | 2.23 | 1.3625 | [1.3136, 1.3732] | 70.53% | FAIL |
| gqa2 s128 | local2 | 2.49 | 1.8065 | [1.7865, 1.8332] | 45.88% | FAIL |
| mha4 s4 | local2 | 2.1 | 1.2784 | [1.1873, 1.3463] | 74.69% | FAIL |
| mha4 s128 | local2 | 2.91 | 1.7961 | [1.5702, 2.3186] | 58.32% | FAIL |

达成 **0/4** 格，要求至少 3 格。默认放置四格端到端耗时几何平均最小的配置为 **baseline**。

| 配置 | 达门格数 | 协议/barrier 几何平均 | full L2 几何平均 ms |
|---|---|---|---|
| baseline | 0/4 | 2.4243 | 0.7129 |
| c1 | 0/4 | 2.5712 | 0.7174 |
| c2 | 0/4 | 2.6019 | 0.7180 |
| window2 | 0/4 | 2.9262 | 0.7595 |
| local2 | 0/4 | 1.5418 | 0.7367 |
| window4 | 0/4 | 2.9759 | 0.7613 |
| local4 | 0/4 | 1.5750 | 0.7386 |

若研究门代表配置与端到端最快配置不同，两者分别报告；窗口配置均为显式实验设置，没有据此改变 W=1 默认值。

## 4. 逐机制消融

单位均为 ms。`baseline` 是 R3 B（E3-0..3 全开）；`c1` 增加单发布者 fence；`c2` 再增加 warp 发布；`windowN` 为 C1+C2、W=N；`localN` 再开 shared 完成标志。`l1no` 是 l1nosync 臂的 L1 耗时，其他探针列为 L2；差分中位数并不要求等于各中位数之差。完整 CI 在 [ablation.tsv](ablation.tsv)。

| 配置 | cell | 放置 | full | nofence | nowait | neither | l1no | fence | wait | notify | barrier | 协议/barrier | L2/L1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | gqa2 s4 | 0 | 0.4229 | 0.4086 | 0.3891 | 0.3533 | 0.3645 | 0.0152 | 0.0348 | 0.0348 | 0.0268 | 2.5769 | 1.0810 |
| baseline | gqa2 s4 | 5 | 0.2846 | 0.2703 | 0.0625 | 0.0594 | 0.3656 | 0.0143 | 0.2222 | 0.0031 | 0.0267 | 8.4412 | 0.7240 |
| baseline | gqa2 s128 | 0 | 0.5969 | 0.5743 | 0.5519 | 0.5189 | 0.5263 | 0.0225 | 0.0440 | 0.0338 | 0.0326 | 2.4048 | 1.0678 |
| baseline | gqa2 s128 | 5 | 0.4444 | 0.4278 | 0.1722 | 0.1567 | 0.5263 | 0.0172 | 0.2723 | 0.0154 | 0.0317 | 9.0625 | 0.7963 |
| baseline | mha4 s4 | 0 | 0.8448 | 0.8161 | 0.7793 | 0.7107 | 0.7281 | 0.0299 | 0.0666 | 0.0676 | 0.0535 | 2.4797 | 1.0813 |
| baseline | mha4 s4 | 5 | 0.5622 | 0.5376 | 0.1421 | 0.1352 | 0.7281 | 0.0246 | 0.4201 | 0.0069 | 0.0534 | 7.8598 | 0.7190 |
| baseline | mha4 s128 | 0 | 1.2114 | 1.1397 | 1.0660 | 0.9971 | 1.0383 | 0.0553 | 0.1350 | 0.0735 | 0.0627 | 2.2479 | 1.0833 |
| baseline | mha4 s128 | 5 | 0.8704 | 0.8612 | 0.3288 | 0.2888 | 1.0568 | 0.0451 | 0.5414 | 0.0399 | 0.0092 | 9.8829 | 0.8151 |
| c1 | gqa2 s4 | 0 | 0.4270 | 0.4086 | 0.3901 | 0.3542 | 0.3645 | 0.0181 | 0.0359 | 0.0369 | 0.0265 | 2.7848 | 1.0895 |
| c1 | gqa2 s4 | 5 | 0.2836 | 0.2703 | 0.0615 | 0.0594 | 0.3656 | 0.0135 | 0.2216 | 0.0031 | 0.0268 | 8.3846 | 0.7232 |
| c1 | gqa2 s128 | 0 | 0.5990 | 0.5735 | 0.5540 | 0.5191 | 0.5263 | 0.0246 | 0.0451 | 0.0348 | 0.0317 | 2.5305 | 1.0729 |
| c1 | gqa2 s128 | 5 | 0.4446 | 0.4280 | 0.1731 | 0.1567 | 0.5263 | 0.0174 | 0.2716 | 0.0162 | 0.0318 | 9.0490 | 0.7975 |
| c1 | mha4 s4 | 0 | 0.8500 | 0.8151 | 0.7793 | 0.7107 | 0.7291 | 0.0338 | 0.0697 | 0.0696 | 0.0524 | 2.6275 | 1.0877 |
| c1 | mha4 s4 | 5 | 0.5612 | 0.5376 | 0.1422 | 0.1350 | 0.7290 | 0.0236 | 0.4188 | 0.0071 | 0.0532 | 8.0042 | 0.7176 |
| c1 | mha4 s128 | 0 | 1.2186 | 1.1305 | 1.0906 | 1.0148 | 1.0320 | 0.0633 | 0.1079 | 0.0768 | 0.0780 | 2.3605 | 1.0896 |
| c1 | mha4 s128 | 5 | 0.8899 | 0.8468 | 0.3287 | 0.2888 | 1.0557 | 0.0502 | 0.5765 | 0.0389 | 0.0524 | 9.9180 | 0.8122 |
| c2 | gqa2 s4 | 0 | 0.4270 | 0.4085 | 0.3891 | 0.3543 | 0.3645 | 0.0182 | 0.0367 | 0.0352 | 0.0268 | 2.7056 | 1.0899 |
| c2 | gqa2 s4 | 5 | 0.2856 | 0.2703 | 0.0622 | 0.0594 | 0.3655 | 0.0152 | 0.2234 | 0.0030 | 0.0267 | 8.4743 | 0.7277 |
| c2 | gqa2 s128 | 0 | 0.5990 | 0.5736 | 0.5540 | 0.5183 | 0.5263 | 0.0255 | 0.0451 | 0.0348 | 0.0329 | 2.4544 | 1.0720 |
| c2 | gqa2 s128 | 5 | 0.4453 | 0.4280 | 0.1730 | 0.1567 | 0.5263 | 0.0172 | 0.2724 | 0.0154 | 0.0319 | 9.0060 | 0.7978 |
| c2 | mha4 s4 | 0 | 0.8520 | 0.8153 | 0.7823 | 0.7117 | 0.7281 | 0.0369 | 0.0707 | 0.0727 | 0.0524 | 2.6604 | 1.0916 |
| c2 | mha4 s4 | 5 | 0.5611 | 0.5375 | 0.1422 | 0.1352 | 0.7281 | 0.0236 | 0.4188 | 0.0071 | 0.0543 | 7.8729 | 0.7181 |
| c2 | mha4 s128 | 0 | 1.2196 | 1.1570 | 1.0977 | 1.0086 | 1.0547 | 0.0635 | 0.1014 | 0.0791 | 0.0676 | 2.5942 | 1.0905 |
| c2 | mha4 s128 | 5 | 0.8847 | 0.8447 | 0.3269 | 0.2888 | 1.0568 | 0.0502 | 0.5676 | 0.0399 | 0.0299 | 9.9180 | 0.8143 |
| window2 | gqa2 s4 | 0 | 0.4495 | 0.4361 | 0.4270 | 0.3748 | 0.3647 | 0.0143 | 0.0225 | 0.0522 | 0.0275 | 2.7644 | 1.1464 |
| window2 | gqa2 s4 | 5 | 0.2940 | 0.2806 | 0.0645 | 0.0614 | 0.3656 | 0.0133 | 0.2294 | 0.0029 | 0.0268 | 8.6683 | 0.7496 |
| window2 | gqa2 s128 | 0 | 0.6349 | 0.6154 | 0.5949 | 0.5396 | 0.5263 | 0.0203 | 0.0400 | 0.0553 | 0.0318 | 3.0000 | 1.1379 |
| window2 | gqa2 s128 | 5 | 0.4669 | 0.4516 | 0.1853 | 0.1700 | 0.5262 | 0.0146 | 0.2815 | 0.0162 | 0.0319 | 9.3226 | 0.8364 |
| window2 | mha4 s4 | 0 | 0.8938 | 0.8662 | 0.8499 | 0.7496 | 0.7281 | 0.0276 | 0.0440 | 0.1014 | 0.0532 | 2.7059 | 1.1440 |
| window2 | mha4 s4 | 5 | 0.5878 | 0.5620 | 0.1475 | 0.1475 | 0.7291 | 0.0248 | 0.4403 | 0.0000 | 0.0537 | 8.0516 | 0.7495 |
| window2 | mha4 s128 | 0 | 1.3046 | 1.2227 | 1.1715 | 1.0353 | 1.0559 | 0.0584 | 0.1300 | 0.1178 | 0.0654 | 3.2673 | 1.1651 |
| window2 | mha4 s128 | 5 | 0.9564 | 0.9052 | 0.3615 | 0.3256 | 1.0578 | 0.0389 | 0.5919 | 0.0358 | 0.0573 | 10.4039 | 0.8594 |
| local2 | gqa2 s4 | 0 | 0.4332 | 0.4168 | 0.4147 | 0.3973 | 0.3644 | 0.0173 | 0.0193 | 0.0177 | 0.0276 | 1.3625 | 1.1047 |
| local2 | gqa2 s4 | 5 | 0.2806 | 0.2703 | 0.0635 | 0.0632 | 0.3645 | 0.0102 | 0.2171 | 0.0002 | 0.0275 | 7.9441 | 0.7150 |
| local2 | gqa2 s128 | 0 | 0.6195 | 0.5970 | 0.5796 | 0.5612 | 0.5254 | 0.0225 | 0.0399 | 0.0185 | 0.0320 | 1.8065 | 1.1101 |
| local2 | gqa2 s128 | 5 | 0.4566 | 0.4413 | 0.1856 | 0.1738 | 0.5263 | 0.0145 | 0.2704 | 0.0121 | 0.0317 | 8.9032 | 0.8182 |
| local2 | mha4 s4 | 0 | 0.8622 | 0.8305 | 0.8253 | 0.7926 | 0.7281 | 0.0308 | 0.0366 | 0.0347 | 0.0543 | 1.2784 | 1.1020 |
| local2 | mha4 s4 | 5 | 0.5601 | 0.5407 | 0.1444 | 0.1426 | 0.7291 | 0.0184 | 0.4157 | 0.0011 | 0.0553 | 7.5538 | 0.7147 |
| local2 | mha4 s128 | 0 | 1.2728 | 1.1971 | 1.1592 | 1.1049 | 1.0342 | 0.0624 | 0.1036 | 0.0481 | 0.0788 | 1.7961 | 1.1352 |
| local2 | mha4 s128 | 5 | 0.9196 | 0.8786 | 0.3607 | 0.3356 | 1.0578 | 0.0427 | 0.5704 | 0.0251 | 0.0450 | 9.7377 | 0.8429 |
| window4 | gqa2 s4 | 0 | 0.4516 | 0.4383 | 0.4279 | 0.3748 | 0.3645 | 0.0132 | 0.0247 | 0.0529 | 0.0276 | 2.7905 | 1.1509 |
| window4 | gqa2 s4 | 5 | 0.2941 | 0.2814 | 0.0645 | 0.0625 | 0.3645 | 0.0131 | 0.2294 | 0.0030 | 0.0276 | 8.4519 | 0.7501 |
| window4 | gqa2 s128 | 0 | 0.6367 | 0.6164 | 0.5949 | 0.5396 | 0.5263 | 0.0203 | 0.0420 | 0.0553 | 0.0319 | 3.0231 | 1.1410 |
| window4 | gqa2 s128 | 5 | 0.4700 | 0.4536 | 0.1855 | 0.1700 | 0.5253 | 0.0164 | 0.2836 | 0.0157 | 0.0317 | 9.4194 | 0.8420 |
| window4 | mha4 s4 | 0 | 0.8970 | 0.8714 | 0.8509 | 0.7484 | 0.7281 | 0.0266 | 0.0470 | 0.1034 | 0.0544 | 2.7736 | 1.1490 |
| window4 | mha4 s4 | 5 | 0.5878 | 0.5622 | 0.1475 | 0.1475 | 0.7291 | 0.0255 | 0.4403 | 0.0000 | 0.0543 | 8.1114 | 0.7501 |
| window4 | mha4 s128 | 0 | 1.3025 | 1.2186 | 1.1735 | 1.0547 | 1.0568 | 0.0533 | 0.1270 | 0.1128 | 0.0635 | 3.3517 | 1.1652 |
| window4 | mha4 s128 | 5 | 0.9236 | 0.8806 | 0.3604 | 0.3246 | 1.0578 | 0.0440 | 0.5775 | 0.0358 | 0.0331 | 10.1851 | 0.8646 |
| local4 | gqa2 s4 | 0 | 0.4352 | 0.4198 | 0.4147 | 0.3963 | 0.3656 | 0.0162 | 0.0205 | 0.0184 | 0.0268 | 1.4628 | 1.1091 |
| local4 | gqa2 s4 | 5 | 0.2806 | 0.2703 | 0.0635 | 0.0625 | 0.3656 | 0.0102 | 0.2171 | 0.0003 | 0.0275 | 7.9359 | 0.7154 |
| local4 | gqa2 s128 | 0 | 0.6205 | 0.5990 | 0.5796 | 0.5612 | 0.5262 | 0.0215 | 0.0410 | 0.0184 | 0.0320 | 1.8387 | 1.1118 |
| local4 | gqa2 s128 | 5 | 0.4588 | 0.4434 | 0.1853 | 0.1741 | 0.5254 | 0.0161 | 0.2734 | 0.0113 | 0.0319 | 8.9446 | 0.8232 |
| local4 | mha4 s4 | 0 | 0.8643 | 0.8344 | 0.8263 | 0.7916 | 0.7280 | 0.0301 | 0.0372 | 0.0348 | 0.0545 | 1.3061 | 1.1046 |
| local4 | mha4 s4 | 5 | 0.5601 | 0.5407 | 0.1444 | 0.1423 | 0.7291 | 0.0194 | 0.4158 | 0.0017 | 0.0544 | 7.6665 | 0.7149 |
| local4 | mha4 s128 | 0 | 1.2749 | 1.1940 | 1.1438 | 1.0986 | 1.0271 | 0.0573 | 0.1199 | 0.0472 | 0.0830 | 1.7515 | 1.1377 |
| local4 | mha4 s128 | 5 | 0.9144 | 0.8663 | 0.3615 | 0.3366 | 1.0578 | 0.0420 | 0.5520 | 0.0256 | 0.0206 | 9.3427 | 0.8469 |
| baseline | real s4 | 0 | 5.9350 | 5.6289 | 5.6074 | 5.4129 | 5.5409 | 0.0285 | 0.1151 | 0.3041 | 0.1033 | 2.1372 | 1.0449 |
| c1 | real s4 | 0 | 5.9636 | 6.0662 | 5.9972 | 5.5550 | 5.6634 | 0.0289 | 0.1065 | 0.3768 | 0.0952 | 2.4743 | 1.0546 |
| c2 | real s4 | 0 | 5.9074 | 6.0713 | 5.9698 | 5.6852 | 5.4004 | 0.0164 | 0.0909 | 0.2857 | 0.1230 | 1.6715 | 1.0548 |
| window2 | real s4 | 0 | 5.9176 | 6.3212 | 6.0283 | 5.6863 | 5.4292 | 0.0124 | 0.0451 | 0.5323 | 0.1057 | 2.6791 | 1.0826 |
| local2 | real s4 | 0 | 5.9379 | 5.9105 | 6.0652 | 5.9105 | 5.5357 | 0.0246 | -0.0558 | 0.0642 | 0.1005 | 0.4000 | 1.0357 |

## 5. Fence 定价与 C 的实际设计依据

| cell | 放置 | fence us | notify us | full−neither us | fence/notify | fence/(full−neither) |
|---|---|---|---|---|---|---|
| gqa2 s4 | 0 | 15.3600 | 35.8400 | 69.6320 | 0.4191 | 0.2146 |
| gqa2 s4 | 5 | 13.3760 | 2.0480 | 224.2560 | 6.4098 | 0.0597 |
| gqa2 s128 | 0 | 22.5280 | 33.6640 | 77.8240 | 0.6642 | 0.2857 |
| gqa2 s128 | 5 | 16.4800 | 15.3600 | 287.7440 | 1.0711 | 0.0575 |
| mha4 s4 | 0 | 31.7440 | 66.5600 | 135.1360 | 0.4672 | 0.2316 |
| mha4 s4 | 5 | 24.5760 | 7.1680 | 427.2000 | 3.5392 | 0.0576 |
| mha4 s128 | 0 | 56.1600 | 79.9680 | 187.3920 | 0.7105 | 0.3307 |
| mha4 s128 | 5 | 45.0560 | 40.9600 | 609.2800 | 1.0992 | 0.0730 |

默认放置 fence 仅占协议差分约 21–33%，rotate 约 6–7%；因此没有把 C1 的 128→1 个线程参与误当作整个协议 128 倍缩减。C2 着重让同 warp 的两条独立发布并行，C3(a) 测 shared 状态对已有窗口执行器的实际影响。rotate 的 fence/notify 可大于 1：fence 在 wait 开启时测量，notify 在 wait 关闭时定价，二者存在交互，未截断比值。

## 6. 协议实现、正确性与作用域

C1：NotifyTask 先 CTA barrier，再由 thread 0 fence 后发布。新 litmus 共 1,800 个进程，覆盖 grid=64/128/256、tile=1024/4096；cache 与 delayed-writer 两个独立套件 分别使 no-fence、no-barrier 负对照在所有格 50/50 失败，正对照全部 50/50 通过。旧的 3,600 进程盲区复核保留在 `litmus_recheck/`。§8.5 在新复核通过后才追加解封注记。两模型静态 L2 MEMBAR.SC.GPU 为 2→2；变化是发布 fence 的参与线程 128→1 （常规 128-thread CTA 中发出该指令的 warp 从 4 个降到 1 个），不是静态指令站点消失。`c1/sass/` 与 `../FENCE/raw/sass/` 保存 BARRIERS 段和相邻谓词证据。

C2：warp 0 的 lane 0 fence 后经 syncwarp 把顺序传给发布 lanes；lane 0/1 分别处理细粒度/聚合事件，其他 warp 可进入下一个 slot 的等待。发布 lanes 在自身发布前 不进入下一个依赖等待，因此即使该等待依赖本 task 的未发布事件，也不会形成发布者等待自身 事件的环。下一次 RunTask 前的 CTA 汇合保留。`c2_dependency/trace` 重建 κ=2 且 σ 省略未生效的真实相邻 slot 依赖，正确性原始格由验证器逐个检查。未引入 Prefetch。

C3(a)：窗口内不能省略的同 worker 边在旧实现中已用 `slot_local_deps` 与寄存器 `done_mask` 处理，并不存在等待移除的全局 poll。本轮按目标改为独立于 TaskSmem 的 shared 完成标志，并在 W=2/4 做对应对照。TaskSmem union 生命周期不变；无出事件的 task 也对 shared 标志提供 CTA 汇合。

C3(b)：只把 DSMEM 的 cluster 内 fan-in 到达降为 cluster scope；最后一个本地 到达向跨 cluster 消费者发布时仍保留 GPU fence。修复 RED 与 shard 的组合后，消费者等待非空 shard 数，而非原始 writer 数。sm_89 的组合正确性、caps=false SASS 退化及 sm_120 PTX/完整模型编译有证据；sm_120 硬件正确性与性能未运行，不能据 sm_89 推定通过。

| sm_120 runner | NEED_MIB | CPU 自检 | 目标机任务 |
|---|---|---|---|
| [FENCE/run_sm120.sh](../FENCE/run_sm120.sh) | 4096 | PASS | 两放置五臂定价 |
| [SYNC_V3/run_sm120.sh](run_sm120.sh) | 16384 | PASS | C1/C2/C3(a) 消融、cluster off/on、real-width |
| [CHAIN2/run_sm120.sh](../CHAIN2/run_sm120.sh) | 16384 | PASS | 独立 hop/BF16 标定、重解 Plan、四臂链化对照 |

三个 runner 均未在 sm_120 上运行。目标机器需重建 build-portable，并准备参考模型与 real-width 的可移植 source/export/fixture；不得搬运 sm_89 的物化 worker/slot 表。CHAIN2 的新 hop 曲线写入自己的 calibration 目录，历史 sm_120 证据保持只读。

| 配置 | 模型 | 放置 | 静态 MEMBAR.SC.GPU | 静态 BAR.SYNC |
|---|---|---|---|---|
| baseline | gqa2 | 0 | 2 | 8 |
| baseline | gqa2 | 5 | 2 | 8 |
| baseline | mha4 | 0 | 2 | 8 |
| baseline | mha4 | 5 | 2 | 8 |
| c1 | gqa2 | 0 | 2 | 8 |
| c1 | gqa2 | 5 | 2 | 8 |
| c1 | mha4 | 0 | 2 | 8 |
| c1 | mha4 | 5 | 2 | 8 |
| c2 | gqa2 | 0 | 2 | 8 |
| c2 | gqa2 | 5 | 2 | 8 |
| c2 | mha4 | 0 | 2 | 8 |
| c2 | mha4 | 5 | 2 | 8 |
| window2 | gqa2 | 0 | 3 | 8 |
| window2 | gqa2 | 5 | 3 | 8 |
| window2 | mha4 | 0 | 3 | 8 |
| window2 | mha4 | 5 | 3 | 8 |
| local2 | gqa2 | 0 | 3 | 10 |
| local2 | gqa2 | 5 | 3 | 10 |
| local2 | mha4 | 0 | 3 | 10 |
| local2 | mha4 | 5 | 3 | 10 |
| window4 | gqa2 | 0 | 3 | 8 |
| window4 | gqa2 | 5 | 3 | 8 |
| window4 | mha4 | 0 | 3 | 8 |
| window4 | mha4 | 5 | 3 | 8 |
| local4 | gqa2 | 0 | 3 | 10 |
| local4 | gqa2 | 5 | 3 | 10 |
| local4 | mha4 | 0 | 3 | 10 |
| local4 | mha4 | 5 | 3 | 10 |

## 7. 修正归因、旧界与窗口回退

### R4 task-DAG reconstruction audit

✅ Verified offline from the 32 R3 placement traces and 12 R3 window traces. No historical file was changed. Input paths and SHA256 digests are in `manifest.tsv` and `inputs_sha256.tsv`. These are reanalyses of historical measurements, not new R4 kernel timings.

The three historical dump tables do not contain the unsimplified DAG: `schedule.tsv` contains offsets, `waits.tsv` has already lost local and lifted waits, and `events.tsv` identifies publication rows. Reconstruction therefore reads the matching generated `kDependencies0` table, expands its runtime windows, and checks each traced task's dependency slice. Trace `(worker, slot, stage, logical_task)` supplies materialized sigma.

All original analyzer metrics remain available with `_legacy` suffixes. The corrected zero-sync DAG bound includes all producer nodes even on zero-cost same-worker edges. Explicit executor edges are distinct: at W=1 the preceding slot must complete; at W>1 every slot at least W positions back must complete. These edges enter the separate `cp_plan_nosync_ns`, not the placement-independent task-DAG bound. Actual execution-order edges are used only for causal reconstruction.

Zero-globaltimer-duration nodes remain on tied longest paths (maximize node count, then break ties by task coordinates). This preserves meaningful node counts without assigning invented durations to sub-tick work. Cycles and unknown predecessors are rejected.

#### A-a / A-b / A-c

44/44 satisfy measured >= max(corrected DAG bound, busiest-worker work), including all 32 required placement dumps. 44/44 causal interval partitions close exactly, with nonnegative gap and wait <= kernel span. gqa2 s4 A rotate and chain both have 20 nodes and 242688 ns task weight.

For the split, each predecessor-to-successor interval starts at the predecessor's run_end. Publish gets its measured portion first, then only the non-overlapping remainder of the successor wait and pre-run barrier is counted. The final node's publication tail is included. Root pre-run delay is excluded and the start anchor is the root run_begin. Thus `cp_chain_span_ns` and the reconstruction have the same endpoints. No negative values are clamped to make the partition pass.

#### A-d: rotate under A/B/D/W

| model | seq | config | legacy DAG bound us | corrected DAG bound us | corrected reconstruction us | trace span us |
|---|---:|---|---:|---:|---:|---:|
| gqa2 | 128 | a | 89.088 | 346.112 | 454.656 | 455.680 |
| gqa2 | 4 | a | 242.688 | 242.688 | 286.720 | 287.744 |
| mha4 | 128 | a | 91.136 | 697.344 | 948.224 | 948.224 |
| mha4 | 4 | a | 244.736 | 488.448 | 572.416 | 572.416 |
| gqa2 | 128 | b | 90.112 | 349.184 | 441.344 | 441.344 |
| gqa2 | 4 | b | 242.688 | 242.688 | 282.624 | 282.624 |
| mha4 | 128 | b | 90.112 | 695.296 | 909.312 | 909.312 |
| mha4 | 4 | b | 245.760 | 491.520 | 567.296 | 568.320 |
| gqa2 | 128 | d | 115.712 | 351.232 | 462.848 | 462.848 |
| gqa2 | 4 | d | 250.880 | 250.880 | 289.792 | 289.792 |
| mha4 | 128 | d | 206.848 | 710.656 | 972.800 | 973.824 |
| mha4 | 4 | d | 253.952 | 507.904 | 586.752 | 586.752 |
| gqa2 | 128 | w | 112.640 | 342.016 | 466.944 | 467.968 |
| gqa2 | 4 | w | 244.736 | 244.736 | 286.720 | 286.720 |
| mha4 | 128 | w | 203.776 | 708.608 | 993.280 | 994.304 |
| mha4 | 4 | w | 247.808 | 493.568 | 579.584 | 579.584 |

The corrected bound does not consult worker assignment, wait counts, or W. Its remaining differences across traces are different observed node durations (including 1024-ns quantization and configuration-dependent tracing/CTA execution), not disappearance of DAG edges. The new regression test holds one trace fixed and varies W=1/2/4, obtaining the same DAG bound. No duration renormalization or placement-specific adjustment is applied.

#### Window reassessment

| model | seq | W | legacy cp us | corrected cp us | legacy HOL us | corrected HOL us |
|---|---:|---:|---:|---:|---:|---:|
| gqa2 | 128 | 1 | 187.392 | 324.608 | 79889.408 | 22872.064 |
| gqa2 | 4 | 1 | 136.192 | 232.448 | 1274.880 | 483.328 |
| mha4 | 128 | 1 | 196.608 | 685.056 | 207584.256 | 47318.016 |
| mha4 | 4 | 1 | 136.192 | 463.872 | 7692.288 | 1936.384 |
| gqa2 | 128 | 2 | 338.944 | 342.016 | 86979.584 | 14131.200 |
| gqa2 | 4 | 2 | 245.760 | 246.784 | 1347.584 | 314.368 |
| mha4 | 128 | 2 | 679.936 | 684.032 | 217905.152 | 44335.104 |
| mha4 | 4 | 2 | 482.304 | 489.472 | 8433.664 | 1215.488 |
| gqa2 | 128 | 4 | 337.920 | 342.016 | 86464.512 | 2158.592 |
| gqa2 | 4 | 4 | 239.616 | 242.688 | 1360.896 | 62.464 |
| mha4 | 128 | 4 | 681.984 | 688.128 | 216441.856 | 23089.152 |
| mha4 | 4 | 4 | 483.328 | 486.400 | 8445.952 | 185.344 |

The corrected HOL estimate includes every semantic predecessor when testing future readiness. It counts only the part of a head stall after a later task actually becomes data-ready. Whole waits are no longer counted merely because some later slot has an empty, already-lifted wait interval. W=4 reduces this estimate in all four window cells; the old conclusion that the window reclaimed no HOL is not supported by the corrected analysis. HOL is summed across workers and must not be added directly to kernel elapsed time.

##### Original paired timing is independent of the analysis formula

| model | seq | paired W2/A ratio median | ratio of medians | raw pairs |
|---|---:|---:|---:|---:|
| gqa2 | 4 | 1.035531 | 1.037915 | 25 |
| gqa2 | 128 | 1.044141 | 1.044843 | 25 |
| mha4 | 4 | 1.038869 | 1.038651 | 25 |
| mha4 | 128 | 1.047805 | 1.047619 | 25 |

The analyzer does not produce E2E_TIME. Its repair removes **0% of the raw paired timing regression**: 100% of that recorded end-to-end delta remains. This does not quantify how much is hardware noise versus scan/probe/fence overhead. A new paired experiment is necessary for that causal allocation; no physical-overhead percentage is invented here. The measurement artifact was in the inferred lower bound and HOL interpretation, not in the arithmetic that timed the kernel. W remains disabled by default.

#### Limits

Only rotate and chain have the 32 historical placement dumps. E needs new A traces for the other candidates before a complete freeze. Exact-ISL runtime descriptors fail explicitly in this parser; they need an authoritative task-DAG export. No poll-derived fallback is presented as a corrected DAG.

补充：上述 32 个历史 dump 只有 rotate/chain；其余四候选的新 A trace 已由 `targets_raw/` 补齐后完成 24 项冻结。

本轮 shared 标志对窗口的直接改善（差值均为 local−control）：

| cell | W | local/control | 95% CI | 耗时改善 | full 差 us | neither 差 us | local/C2(W=1) 与 CI |
|---|---|---|---|---|---|---|---|
| gqa2_s4_p0 | 2 | 0.9636 | [0.9635, 0.9657] | 3.64% | -16.3840 | 22.3680 | 1.0156 [1.0120, 1.0169] |
| gqa2_s4_p0 | 4 | 0.9637 | [0.9614, 0.9659] | 3.63% | -16.4160 | 21.5040 | 1.0210 [1.0191, 1.0219] |
| gqa2_s4_p5 | 2 | 0.9545 | [0.9514, 0.9549] | 4.55% | -13.3760 | 1.0240 | 0.9821 [0.9817, 0.9847] |
| gqa2_s4_p5 | 4 | 0.9544 | [0.9516, 0.9549] | 4.56% | -13.4400 | 0.1600 | 0.9821 [0.9819, 0.9857] |
| gqa2_s128_p0 | 2 | 0.9758 | [0.9743, 0.9758] | 2.42% | -15.3600 | 21.6320 | 1.0342 [1.0336, 1.0342] |
| gqa2_s128_p0 | 4 | 0.9743 | [0.9739, 0.9758] | 2.57% | -16.3840 | 21.4400 | 1.0359 [1.0342, 1.0363] |
| gqa2_s128_p5 | 2 | 0.9781 | [0.9778, 0.9799] | 2.19% | -10.2400 | 4.0960 | 1.0251 [1.0248, 1.0259] |
| gqa2_s128_p5 | 4 | 0.9782 | [0.9760, 0.9782] | 2.18% | -10.2400 | 4.0960 | 1.0319 [1.0299, 1.0323] |
| mha4_s4_p0 | 2 | 0.9645 | [0.9623, 0.9658] | 3.55% | -31.7120 | 44.0320 | 1.0117 [1.0096, 1.0130] |
| mha4_s4_p0 | 4 | 0.9634 | [0.9613, 0.9646] | 3.66% | -32.8000 | 43.0080 | 1.0142 [1.0120, 1.0145] |
| mha4_s4_p5 | 2 | 0.9529 | [0.9528, 0.9545] | 4.71% | -27.6480 | -4.9920 | 0.9980 [0.9964, 0.9982] |
| mha4_s4_p5 | 4 | 0.9529 | [0.9513, 0.9546] | 4.71% | -27.6480 | -5.1200 | 0.9997 [0.9964, 1.0014] |
| mha4_s128_p0 | 2 | 0.9749 | [0.9726, 0.9795] | 2.51% | -32.8640 | 72.6080 | 1.0429 [1.0277, 1.0543] |
| mha4_s128_p0 | 4 | 0.9771 | [0.9742, 0.9834] | 2.29% | -29.8560 | 27.3920 | 1.0463 [1.0420, 1.0543] |
| mha4_s128_p5 | 2 | 0.9786 | [0.9504, 1.0034] | 2.14% | -20.6080 | 11.2640 | 1.0381 [1.0087, 1.0596] |
| mha4_s128_p5 | 4 | 0.9736 | [0.9501, 1.0256] | 2.64% | -24.5760 | 11.4240 | 1.0344 [1.0011, 1.0643] |

必须区分完整 kernel 的改善与协议差分的改变：shared 完成状态即使在 neither 臂也需要 CTA 汇合，否则其他 warp 可能读取尚未更新的 head/done。W=2 的 neither 原始 SASS 在两模型上均为 register 6 个、shared 8 个 BAR.SYNC 位点（`local_probe_sass/`）。这些本地同步成本保留在 neither 中；full−neither 因而不能把本地状态管理也计作 全局 wait/notify。研究门仍原样计算，但其差距缩减不能全部解释为端到端加速，上表独立展示两臂的实际变化。参考模型的原始 E2E_RESOURCE 同时显示，register/shared 窗口均为 2 CTA/SM、grid=256；寄存器数 218→220，静态 shared memory 0→16 bytes，TaskSmem 仍为 24576 bytes。没有通过改变 驻留 grid 来获得这组 local/control 差异。

## 8. 冻结 target 与本轮位置

完整冻结表：[targets.tsv](targets.tsv)，原始 trace 与 SHA 在 `targets_raw/`。每项 floor、measured_A 和 target 均从该候选配置 A 的同一观测上下文得到；24/24 target 不低于自己的 floor。floor 使用有 trace 的 task 时间，端到端位置使用 全新无 trace 进程，因此低于观测 floor 不等于突破硬件理论下限。下表每个候选只比较冻结的原 Plan，新增 Chain2 不借用旧 chain 的 floor。

| 候选 | cell | floor ms | measured_A ms | target ms | 最优 W=1 协议 | 本轮 L2 ms | 相对 target ms |
|---|---|---|---|---|---|---|---|
| legacy_grid_stride | gqa2 s4 | 0.3666 | 0.4535 | 0.4100 | baseline | 0.4229 | 0.0129 |
| balanced | gqa2 s4 | 0.2775 | 0.6482 | 0.4628 | baseline | 0.5849 | 0.1220 |
| rotate | gqa2 s4 | 0.2427 | 0.2918 | 0.2673 | c1 | 0.2836 | 0.0164 |
| eft | gqa2 s4 | 0.2468 | 0.2992 | 0.2730 | baseline | 0.2796 | 0.0065 |
| wavefront | gqa2 s4 | 0.2427 | 0.3011 | 0.2719 | baseline | 0.2816 | 0.0097 |
| chain | gqa2 s4 | 0.2437 | 0.3000 | 0.2719 | baseline | 0.2792 | 0.0073 |
| legacy_grid_stride | gqa2 s128 | 0.5315 | 0.6246 | 0.5780 | baseline | 0.5969 | 0.0188 |
| balanced | gqa2 s128 | 0.3359 | 1.0527 | 0.6943 | baseline | 0.9062 | 0.2120 |
| rotate | gqa2 s128 | 0.3492 | 0.4638 | 0.4065 | baseline | 0.4444 | 0.0379 |
| eft | gqa2 s128 | 0.3840 | 0.4669 | 0.4255 | baseline | 0.4393 | 0.0138 |
| wavefront | gqa2 s128 | 0.3820 | 0.4668 | 0.4244 | baseline | 0.4393 | 0.0149 |
| chain | gqa2 s128 | 0.3492 | 0.5284 | 0.4388 | baseline | 0.4934 | 0.0547 |
| legacy_grid_stride | mha4 s4 | 0.7455 | 0.9010 | 0.8232 | baseline | 0.8448 | 0.0216 |
| balanced | mha4 s4 | 0.4915 | 1.2370 | 0.8643 | baseline | 1.0148 | 0.1506 |
| rotate | mha4 s4 | 0.4884 | 0.5806 | 0.5345 | c2 | 0.5611 | 0.0266 |
| eft | mha4 s4 | 0.4905 | 0.5960 | 0.5432 | baseline | 0.5099 | -0.0333 |
| wavefront | mha4 s4 | 0.4925 | 0.5981 | 0.5453 | baseline | 0.5128 | -0.0325 |
| chain | mha4 s4 | 0.4956 | 0.5950 | 0.5453 | baseline | 0.5080 | -0.0373 |
| legacy_grid_stride | mha4 s128 | 1.0742 | 1.2769 | 1.1756 | baseline | 1.2114 | 0.0358 |
| balanced | mha4 s128 | 0.8858 | 4.1113 | 2.4985 | baseline | 3.2860 | 0.7875 |
| rotate | mha4 s128 | 0.6267 | 0.8663 | 0.7465 | baseline | 0.8704 | 0.1239 |
| eft | mha4 s128 | 0.7209 | 0.8919 | 0.8064 | baseline | 0.8366 | 0.0302 |
| wavefront | mha4 s128 | 0.7311 | 0.9605 | 0.8458 | baseline | 0.8202 | -0.0256 |
| chain | mha4 s128 | 0.6482 | 1.1295 | 0.8888 | baseline | 1.0332 | 0.1444 |

## 9. 代价感知链化

### 实现与对照

All switches remain off by default. `ChainRequest::cost_aware_extend` rejects
an edge unless its calibrated hop is strictly greater than the successor's
`task_ns`. The calibration is `SIMULATOR/hop_ns.tsv` evaluated at the producer's
fan-out, approximately 1234 ns on sm_89, rather than a newly invented literal.
The successor uses the same cost-model weight as chain extraction. Equality
rejects. Existing queue capacity formulas and hard legality checks are retained.

The simple price test alone does not satisfy D-b. `minimal_variant.cpp` is a
reproducible isolated diagnostic containing that test without the final fill
changes; `build_variant.py` links it without replacing the shared library.
The final implementation also ranks ready tasks by remaining critical work,
prices sibling CTA sharing on the same SM in `finish_on`, and breaks equal
finish-time ties toward fewer path hops. The selected experiment uses four
**existing** feedback rounds to carry realized queue blocking into extraction.
These are explicit changes to the initial D recipe, authorized by the user's
instruction to solve a mistaken premise rather than stop all R4 work.
`design.json` froze this recipe before any Chain2 GPU timing.

The GPU comparison has four arms: rotate; the completed cost-aware recipe;
the same four-feedback-round recipe with cost awareness disabled; and the
unchanged R3 chain with feedback disabled. All use configuration A so the
measured protocol matches the hop calibration. The matched control separates
the feedback setting from the cost-aware implementation. No `ChainDP` code is
changed and no window becomes default.

`final/replay/path/*_path.tsv` records each simulator critical-path node and
edge. The verifier counts cross-worker edges directly, checks edge types
against worker IDs and requires replayed chain source bytes to match the
measured materialized source. It does not accept `predicted.tsv` as a D-b
verdict. The replay uses the same local geometry and manifest as the final
solver run. Raw edge-price exclusions are stored in exact-price histogram
buckets in `rejected_extensions.tsv`; their counts are unique DAG edges in
the selected extraction pass, not repeated DP visits. Each bucket records
hop, queue cost, multiplicity and an example producer/successor.

The sm_120 runner regenerates geometry, BF16 calibration, the RMW/load hop
curve and all materialized Plans on its target. The curve is fitted into
its own `calibration/` directory; the old `SIMULATOR/raw_sm120` evidence is
neither reused as a new measurement nor overwritten. CPU self-check refits
the committed sm_89 sweeps and verifies the exact existing coefficients.
Its original/feedback-matched/cost-aware arms expose
whether a cheaper hop changes the relative value of the extension test.
Only CPU self-checks and sm_120 compilation have been performed here.

| cell | 臂 | L2 ms | 对 rotate 配对比 | 95% CI |
|---|---|---|---|---|
| gqa2 s4 | rotate | 0.2908 | 1.0000 | [1.0000, 1.0000] |
| gqa2 s4 | chain | 0.2916 | 1.0001 | [1.0000, 1.0033] |
| gqa2 s4 | chain_control | 0.2926 | 1.0036 | [1.0030, 1.0067] |
| gqa2 s4 | original | 0.2929 | 1.0067 | [1.0041, 1.0075] |
| gqa2 s128 | rotate | 0.4557 | 1.0000 | [1.0000, 1.0000] |
| gqa2 s128 | chain | 0.5009 | 1.0993 | [1.0989, 1.1011] |
| gqa2 s128 | chain_control | 0.5149 | 1.1299 | [1.1282, 1.1303] |
| gqa2 s128 | original | 0.5181 | 1.1371 | [1.1369, 1.1371] |
| mha4 s4 | rotate | 0.5765 | 1.0000 | [1.0000, 1.0000] |
| mha4 s4 | chain | 0.5755 | 0.9997 | [0.9977, 1.0081] |
| mha4 s4 | chain_control | 0.5816 | 1.0138 | [1.0110, 1.0159] |
| mha4 s4 | original | 0.5499 | 1.0122 | [1.0104, 1.0140] |
| mha4 s128 | rotate | 0.8550 | 1.0000 | [1.0000, 1.0000] |
| mha4 s128 | chain | 0.9677 | 1.1312 | [1.1304, 1.1329] |
| mha4 s128 | chain_control | 1.0772 | 1.2599 | [1.2585, 1.2613] |
| mha4 s128 | original | 1.0762 | 1.2580 | [1.2563, 1.2601] |
| real s4 | rotate | 4.4072 | 1.0000 | [1.0000, 1.0000] |
| real s4 | chain | 5.1078 | 1.1600 | [1.1588, 1.1687] |
| real s4 | chain_control | 4.5005 | 1.0214 | [1.0205, 1.0237] |
| real s4 | original | 4.5005 | 1.0211 | [1.0205, 1.0232] |
| real s128 | rotate | 6.4993 | 1.0000 | [1.0000, 1.0000] |
| real s128 | chain | 7.1638 | 1.1021 | [1.1019, 1.1024] |
| real s128 | chain_control | 7.3646 | 1.1329 | [1.1326, 1.1335] |
| real s128 | original | 7.3636 | 1.1328 | [1.1324, 1.1332] |

价格测试单独诊断（保留同样四轮反馈）：mha4 s128 路径为 40 跳，模拟 makespan 597.674 us。它尚未满足 D-b，不能作为交付方案；最终队列放置修正与反馈组合的原始路径在下表独立给出。

| cell | 候选 | critical path hops | queue edges | 路径节点 | 路径 task us | 模拟 makespan us |
|---|---|---|---|---|---|---|
| gqa2 s4 | legacy_grid_stride | 17 | 8 | 26 | 289.633 | 310.6165 |
| gqa2 s4 | balanced | 17 | 39 | 57 | 359.430 | 380.4120 |
| gqa2 s4 | rotate | 19 | 0 | 20 | 179.817 | 203.2686 |
| gqa2 s4 | eft | 17 | 0 | 20 | 179.817 | 200.7962 |
| gqa2 s4 | wavefront | 19 | 0 | 20 | 179.817 | 203.2386 |
| gqa2 s4 | chain | 17 | 0 | 20 | 179.817 | 200.7987 |
| gqa2 s128 | legacy_grid_stride | 17 | 10 | 28 | 294.216 | 323.5266 |
| gqa2 s128 | balanced | 19 | 136 | 156 | 298.787 | 325.9113 |
| gqa2 s128 | rotate | 17 | 10 | 28 | 185.816 | 215.1266 |
| gqa2 s128 | eft | 17 | 10 | 28 | 185.816 | 215.1205 |
| gqa2 s128 | wavefront | 17 | 10 | 28 | 185.816 | 215.1237 |
| gqa2 s128 | chain | 17 | 20 | 38 | 189.425 | 218.4992 |
| mha4 s4 | legacy_grid_stride | 35 | 16 | 52 | 579.266 | 622.4502 |
| mha4 s4 | balanced | 31 | 94 | 126 | 710.714 | 748.9787 |
| mha4 s4 | rotate | 39 | 0 | 40 | 359.634 | 407.7692 |
| mha4 s4 | eft | 35 | 0 | 40 | 359.634 | 402.8319 |
| mha4 s4 | wavefront | 39 | 0 | 40 | 359.634 | 407.7315 |
| mha4 s4 | chain | 36 | 0 | 40 | 359.634 | 404.2983 |
| mha4 s128 | legacy_grid_stride | 35 | 28 | 64 | 590.304 | 652.9604 |
| mha4 s128 | balanced | 36 | 518 | 555 | 1189.595 | 1244.4321 |
| mha4 s128 | rotate | 35 | 32 | 68 | 374.453 | 437.1237 |
| mha4 s128 | eft | 36 | 35 | 72 | 375.402 | 437.1238 |
| mha4 s128 | wavefront | 35 | 32 | 68 | 374.453 | 437.1170 |
| mha4 s128 | chain | 35 | 55 | 91 | 384.009 | 446.3261 |
| real s4 | legacy_grid_stride | 35 | 16 | 52 | 5437.812 | 5480.9586 |
| real s4 | balanced | 31 | 448 | 480 | 47721.972 | 47760.2057 |
| real s4 | rotate | 39 | 0 | 40 | 3718.801 | 4340.9013 |
| real s4 | eft | 35 | 2 | 42 | 4005.309 | 4335.9953 |
| real s4 | wavefront | 39 | 0 | 40 | 3718.801 | 4340.9022 |
| real s4 | chain | 37 | 8 | 47 | 4722.158 | 4912.0468 |
| real s128 | legacy_grid_stride | 31 | 184 | 216 | 5647.690 | 5901.1656 |
| real s128 | balanced | 35 | 1063 | 1099 | 11932.036 | 12057.6097 |
| real s128 | rotate | 35 | 168 | 204 | 3928.649 | 4760.0490 |
| real s128 | eft | 35 | 184 | 220 | 4536.554 | 4760.0492 |
| real s128 | wavefront | 35 | 168 | 204 | 3928.649 | 4760.0604 |
| real s128 | chain | 35 | 204 | 240 | 4539.025 | 4929.5809 |

| cell | 拒绝的唯一边数 | hop ns 范围 | queue ns 范围 |
|---|---|---|---|
| gqa2 s4 | 412 | 1233.774–1234.481 | 1584.633–30235.103 |
| mha4 s4 | 908 | 1233.774–1234.481 | 1584.633–30235.103 |
| real s4 | 25952 | 1232.281–1234.481 | 3291.275–485456.747 |
| gqa2 s128 | 538124 | 1231.804–1235.011 | 1496.685–30235.103 |
| mha4 s128 | 2127388 | 1231.804–1235.011 | 1496.685–30235.103 |
| real s128 | 34394336 | 1230.603–1235.011 | 2907.628–485456.042 |

## 10. 偏离与理由

1. 原 H4 提前提交保留历史，恢复阶段重新按 A→E→B→C 排序，详见 §1。
2. dump 的 waits 已经丢失被省略的 DAG 边，因此读取匹配生成源的 `kDependencies0` 补齐图，不能仅凭三个物化表逆推出不存在的信息。
3. R3 RED 实际是 relaxed 原子，序关系仍来自 NotifyTask fence；PTX 原始审计保留。
4. 原 no-barrier 探针对部分几何不敏感；固定 delayed-writer 与独立 cache 套件重新验证，不改期望结果，不把盲区当作同步冗余。
5. C1 改变动态线程参与而非静态 MEMBAR 条数；按真实 SASS 报告。
6. R3 B 已允许非发布 warp 前进；C2 实现实际发布 warp 特化与两条独立发布，没有重复删除已不存在的 barrier。
7. W>1 本地依赖已在寄存器完成跟踪，本轮测量 shared 替换的实际成本。
8. cluster 实验需要先修复 RED/shard 组合；保留 GPU 范围的跨 cluster 转发。
9. D 的价格测试不足以通过跳数门，增加同 SM 队列代价、关键路径排序与跳数平局处理，使用既有反馈四轮；匹配对照及价格测试单独诊断保留。
10. 窗口 no-wait 探针原先仍做 ProbeTaskDependencies 轮询；本轮修复后保留未完成的旧矩阵，并把全部七组配置重新放入一个新会话。正常窗口的完整 SASS 位同，未改变门口径。
11. final SASS 用被测父提交与仅证据子提交绑定，避免要求提交存储自身 hash 的循环。

## 11. 排除项与不变量

EX-E4 预取、EX-E5、EX-S1c、EX-S3、EX-S4、EX-S5、EX-V1 均未实现；未修改 ChainDP、Plan dialect 语义或 TaskSmem union 生命周期；L-a/L-b/L-c/L-e 仍为硬校验。所有新机制默认关闭，W 默认仍为 1，epoch 单调且不在 并发迭代间清零。skeleton 仅在 §5.5.1、§5.5.2、§8.5 追加 v2.1 注记。原有 PLACE_EFT2 summary 与 SYNC_V2 SASS meta 的工作区改动未纳入本轮提交。

## 12. 未达项定位与下一步

发布 fence 只覆盖 B 协议差分的一部分。C1 的 `NotifyTask` 仍把 device fence 放在 CTA barrier 之后、全局计数更新之前，减少参与 warp 并没有删除发布关键路径上的 这个 fence 位点。应先用发布阶段时间戳给 barrier 后的完成延迟定价，再验证显式 global-space release 原子能否合并该序关系；须保留敏感 litmus，避免再次把 STRONG.GPU 当作 release 的证明。完成 C1 后，`WaitTaskDependencies` 仍在非空 wait 后由所有线程执行 acquire fence，`EventPoll` 仍读取全局事件行；`ArriveEvent` 仍需要跨 CTA/跨 cluster 可见的 全局计数更新。当前 EVENT_LOAD_POLL=0 仍以 atomicAdd(ev,0) 做 RMW 轮询；其既有 load 开关须在 WAIT_POLICY=1 的真实调用路径下重新消融（F-162），先用 SASS 确认指令不同，再验证正确性。SOLO 的直接 epoch 路径也被 RED 组合屏蔽，单成员 arrivals 的直接发布仍是可明确验证的后续方案。这些成本随 Plan 的跨 worker 边、event fan-in 与等待数量变化，不能靠继续缩减已经剩一个线程的发布 fence 清除。下一步先给 acquire fence 单独定价，再为协作 acquire 构造带缓存复用与逐 warp 消费的敏感 litmus；不直接把 acquire 也改成 thread 0。并在联合搜索中同时定价 κ 粗化减少事件数与增加就绪等待的代价。

窗口的 `ProbeTaskDependencies` 每探一个候选仍做一次 CTA-wide `__syncthreads_and`，`WindowAcquireSlot` 还要扫描 wait 列表。shared 完成标志 不会消除这些扫描；若 local/control 未改善，就应按窗口探测次数与非阻塞轮询 指令数给扫描定价，再设计批量候选 ready 归约，保持 L-a/L-b/L-c/L-e 与汇合不变量。

链化的成本测试只比较一个 hop 与后继 task 的直接队列成本，尚未完整定价 跨多个同 SM 队列的反向阻塞。原始路径显示 gqa2 s128 的 queue edges 为 10→20，mha4 s128 为 32→55，而 hops 分别仍为 17/35；real s4 的路径 task 时间则从 3718.801 增至 4722.158 us。`free_ns[w] = est_end[node]` 只累计 task 权重，没有单独给仍执行的 NotifyTask 发布成本定价；其对参考格差距的贡献需要下一次 phase trace 验证。`SchedulePass::finish_on` 的 sibling stretch 是近似；对仍慢于 rotate 的格，下一步以模拟器的实际阻塞传播为增量代价，在延长时比较 makespan 增量并回传关键路径边，避免用单节点时间代替整个新增队列边链。sm_120 的更小 hop（约 448 ns）必须在目标机器重新校准和测量，不能由 4090 推定收益。

sm_120 cluster 到达仍待目标硬件运行 `SYNC_V3/run_sm120.sh` 的 cluster_off/on 配对；脚本会在目标上重新求解驻留 Plan。编译通过与 CPU 自检 没有替代该正确性/性能验证。

## 13. 下一轮优先级

1. **EX-S1c + EX-S3 联合搜索优先**：先完成 EX-S1c 已登记的单 Plan 求值预算，再在 EX-S3 中同时优化 κ、跨 worker 跳数和新增队列延迟。六步实现后剩下的 wait/notify 主要仍是全局事件可见性、acquire、计数更新与 Plan 引入的依赖距离；减少真正必要的全局事件数量比继续改发布线程数更直接。
2. **EX-S5 ISL 参数化放置其次**：用于表达并搜索 co-location/跨 stage 归属，但必须把物化队列与同 SM 竞争纳入代价，不能重演“被替换的队列边免费”。
3. **EX-E4 预取再次**：本轮未加入预取。先由新五臂数据确认剩余 task-only 部分与可重叠范围，再衔接 C2 的非发布 warp；必须单独处理 TaskSmem union 的缓存生命周期，不能把当前等待阶段直接改成未验证的预取。
