// E3 comparison baseline: non-persistent version. One tile block per logical
// tile (no loop) -- launched with grid = total number of tiles.
cuda_tile.module @cuda_tile_module {
  entry @non_persistent(%out: tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %out_view = make_tensor_view %out, shape = [1000000], strides = [1] : tensor_view<1000000xf32, strides=[1]>
    %out_pview = make_partition_view %out_view : partition_view<tile=(1), tensor_view<1000000xf32, strides=[1]>>
    %idx_1d = reshape %bx : tile<i32> -> tile<1xi32>
    %idx_f = itof %idx_1d signed : tile<1xi32> -> tile<1xf32>
    %two = constant <f32: 2.000000e+00> : tile<1xf32>
    %val = mulf %idx_f, %two : tile<1xf32>
    %tok = store_view_tko weak %val, %out_pview[%bx] : tile<1xf32>, partition_view<tile=(1), tensor_view<1000000xf32, strides=[1]>>, tile<i32> -> token
    return
  }
}
