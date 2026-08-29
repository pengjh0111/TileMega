// E7: does the currently-built (old) cuda-tile toolchain accept
// `num_worker_warps_per_cta` in optimization_hints, as the newer
// (tensor-ir FetchContent'd) cuda-tile version's AttrDefs.td defines?
cuda_tile.module @cuda_tile_module {
  entry @test_warps(%arg0: tile<ptr<f32>>) optimization_hints=<sm_120 = {num_cta_in_cga = 2, num_worker_warps_per_cta = 4}> {
    return
  }
}
