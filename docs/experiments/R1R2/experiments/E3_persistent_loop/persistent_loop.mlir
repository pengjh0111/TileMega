// E3: persistent kernel using a grid-stride ForOp with runtime SSA bounds.
// Launched with grid = (small, e.g. #SM) tile blocks; each block processes
// many "logical tiles" (total_tiles, a runtime kernel argument) via a
// classic grid-stride loop: for i in (blockId, total_tiles, step=numBlocks).
// This directly tests E1 Q6's finding that ForOp bounds/step can be runtime
// SSA values (not required to be compile-time constants).
cuda_tile.module @cuda_tile_module {
  entry @persistent_loop(%total_tiles: tile<i32>, %out: tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %gx, %gy, %gz = get_num_tile_blocks : tile<i32>
    %c1_i32 = constant <i32: 1> : tile<i32>

    %out_view = make_tensor_view %out, shape = [1000000], strides = [1] : tensor_view<1000000xf32, strides=[1]>
    %out_pview = make_partition_view %out_view : partition_view<tile=(1), tensor_view<1000000xf32, strides=[1]>>

    for %tile_idx in (%bx to %total_tiles, step %gx) : tile<i32> {
      %idx_1d = reshape %tile_idx : tile<i32> -> tile<1xi32>
      %idx_i32 = itof %idx_1d signed : tile<1xi32> -> tile<1xf32>
      %two = constant <f32: 2.000000e+00> : tile<1xf32>
      %val = mulf %idx_i32, %two : tile<1xf32>
      %tok = store_view_tko weak %val, %out_pview[%tile_idx] : tile<1xf32>, partition_view<tile=(1), tensor_view<1000000xf32, strides=[1]>>, tile<i32> -> token
      continue
    }
    return
  }
}
