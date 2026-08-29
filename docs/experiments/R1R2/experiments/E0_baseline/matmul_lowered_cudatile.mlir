module {
  cuda_tile.module @cuda_tile_module {
    entry @matmul_f32_static(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>, %arg2: tile<ptr<f32>>) {
      %cst_0_f32 = constant <f32: 0.000000e+00> : tile<1x64xf32>
      %cst_0_i32 = constant <i32: 0> : tile<i32>
      %cst_128_i32 = constant <i32: 128> : tile<i32>
      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      %0 = remi %blockId_x, %cst_128_i32 unsigned : tile<i32>
      %1 = divi %blockId_x, %cst_128_i32 unsigned : tile<i32>
      %tview = make_tensor_view %arg0, shape = [128, 64], strides = [64, 1] : tensor_view<128x64xf32, strides=[64,1]>
      %pview = make_partition_view %tview : partition_view<tile=(1x64), tensor_view<128x64xf32, strides=[64,1]>>
      %tile, %result_token = load_view_tko weak %pview[%0, %cst_0_i32] : partition_view<tile=(1x64), tensor_view<128x64xf32, strides=[64,1]>>, tile<i32> -> tile<1x64xf32>, token
      %tview_0 = make_tensor_view %arg1, shape = [64, 128], strides = [128, 1] : tensor_view<64x128xf32, strides=[128,1]>
      %pview_1 = make_partition_view %tview_0 : partition_view<tile=(64x64), tensor_view<64x128xf32, strides=[128,1]>>
      %tile_2, %result_token_3 = load_view_tko weak %pview_1[%cst_0_i32, %1] : partition_view<tile=(64x64), tensor_view<64x128xf32, strides=[128,1]>>, tile<i32> -> tile<64x64xf32>, token
      %2 = mmaf %tile, %tile_2, %cst_0_f32 : tile<1x64xf32>, tile<64x64xf32>, tile<1x64xf32>
      %tview_4 = make_tensor_view %arg2, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
      %pview_5 = make_partition_view %tview_4 : partition_view<tile=(1x64), tensor_view<128x128xf32, strides=[128,1]>>
      %3 = store_view_tko weak %2, %pview_5[%0, %1] : tile<1x64xf32>, partition_view<tile=(1x64), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> token
      return
    }
  }
}
