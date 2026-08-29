cuda_tile.module @cuda_tile_module {
  entry @branch_dispatch_n3(%task_type: tile<i32>, %in_a: tile<ptr<f32>>, %in_b: tile<ptr<f32>>, %out: tile<ptr<f32>>) {
    %c0_i32 = constant <i32: 0> : tile<i32>
    %cond0 = cmpi equal %task_type, %c0_i32, signed : tile<i32> -> tile<i1>
    if %cond0 {

      %tview_a0 = make_tensor_view %in_a, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
      %pview_a0 = make_partition_view %tview_a0 : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
      %A0, %tok_a0 = load_view_tko weak %pview_a0[%c0_i32, %c0_i32] : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> tile<128x128xf32>, token
      %tview_b0 = make_tensor_view %in_b, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
      %pview_b0 = make_partition_view %tview_b0 : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
      %B0, %tok_b0 = load_view_tko weak %pview_b0[%c0_i32, %c0_i32] : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> tile<128x128xf32>, token
      %zero_mat0 = constant <f32: 0.000000e+00> : tile<128x128xf32>
      %C0 = mmaf %A0, %B0, %zero_mat0 : tile<128x128xf32>, tile<128x128xf32>, tile<128x128xf32>
      %tview_out0 = make_tensor_view %out, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
      %pview_out0 = make_partition_view %tview_out0 : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
      %tok_out0 = store_view_tko weak %C0, %pview_out0[%c0_i32, %c0_i32] : tile<128x128xf32>, partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> token

      yield
    } else {
      %c1_i32 = constant <i32: 1> : tile<i32>
      %cond1 = cmpi equal %task_type, %c1_i32, signed : tile<i32> -> tile<i1>
      if %cond1 {

      %tview_v1 = make_tensor_view %in_a, shape = [256], strides = [1] : tensor_view<256xf32, strides=[1]>
      %pview_v1 = make_partition_view %tview_v1 : partition_view<tile=(256), tensor_view<256xf32, strides=[1]>>
      %vec1, %tok_v1 = load_view_tko weak %pview_v1[%c0_i32] : partition_view<tile=(256), tensor_view<256xf32, strides=[1]>>, tile<i32> -> tile<256xf32>, token
      %sum1 = reduce %vec1 dim=0 identities=[0.000000e+00 : f32] : tile<256xf32> -> tile<f32>
        (%elem1: tile<f32>, %acc1: tile<f32>) {
          %add1 = addf %elem1, %acc1 : tile<f32>
          yield %add1 : tile<f32>
        }
      %sum_1d1 = reshape %sum1 : tile<f32> -> tile<1xf32>
      %tview_out1 = make_tensor_view %out, shape = [1], strides = [1] : tensor_view<1xf32, strides=[1]>
      %pview_out1 = make_partition_view %tview_out1 : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>
      %tok_out1 = store_view_tko weak %sum_1d1, %pview_out1[%c0_i32] : tile<1xf32>, partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>, tile<i32> -> token

        yield
      } else {

      %tview_a2 = make_tensor_view %in_a, shape = [64, 64], strides = [64, 1] : tensor_view<64x64xf32, strides=[64,1]>
      %pview_a2 = make_partition_view %tview_a2 : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>
      %A2, %tok_a2 = load_view_tko weak %pview_a2[%c0_i32, %c0_i32] : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>, tile<i32> -> tile<64x64xf32>, token
      %tview_b2 = make_tensor_view %in_b, shape = [64, 64], strides = [64, 1] : tensor_view<64x64xf32, strides=[64,1]>
      %pview_b2 = make_partition_view %tview_b2 : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>
      %B2, %tok_b2 = load_view_tko weak %pview_b2[%c0_i32, %c0_i32] : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>, tile<i32> -> tile<64x64xf32>, token
      %zero_mat2 = constant <f32: 0.000000e+00> : tile<64x64xf32>
      %C2 = mmaf %A2, %B2, %zero_mat2 : tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>
      %tview_out2 = make_tensor_view %out, shape = [64, 64], strides = [64, 1] : tensor_view<64x64xf32, strides=[64,1]>
      %pview_out2 = make_partition_view %tview_out2 : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>
      %tok_out2 = store_view_tko weak %C2, %pview_out2[%c0_i32, %c0_i32] : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>, tile<i32> -> token

        yield
      }
    }
    return
  }
}
