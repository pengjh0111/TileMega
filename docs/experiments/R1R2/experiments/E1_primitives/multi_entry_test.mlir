module {
  cuda_tile.module @cuda_tile_module {
    entry @entry_a(%arg0: tile<ptr<f32>>) {
      return
    }
    entry @entry_b(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) {
      return
    }
  }
}
