# V-C — cluster distributed shared memory

Evidence labels: ✅ compiled; ⚠️ documented/not run; ❌ inference.

✅ `cluster_dsmem.cu` contains the complete requested path:
`__cluster_dims__(2,1,1)`, `cooperative_groups::this_cluster()`,
`cluster.map_shared_rank()`, and `cluster.sync()`. It is guarded by the exact
`Caps<Arch>::kCluster` contract, not a preprocessor architecture test.

| Target | Compile | Runtime |
|---|---|---|
| sm_90 | ✅ PASS | ⚠️ not run |
| sm_120 | ✅ PASS | ⚠️ not run |

Runtime requires sm_90+ cluster-capable hardware and was not run on the RTX
4090 development machine.

CUDA documents thread-block clusters as co-scheduled within one GPC, with the
ordinary grid dimension still counting blocks. ⚠️ For a persistent clustered
megakernel, usable capacity must therefore be obtained from
`cudaOccupancyMaxActiveClusters` and rounded to whole cluster shapes; the
ordinary per-kernel CTA formula is insufficient. The GPC distribution and
portable cluster-size limit need to be probed on the migration device.

## Reproduction

```bash
docs/experiments/V_C/run.sh
cat docs/experiments/V_C/raw/crosscompile.txt
```

Documentation: [CUDA programming model — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html).
