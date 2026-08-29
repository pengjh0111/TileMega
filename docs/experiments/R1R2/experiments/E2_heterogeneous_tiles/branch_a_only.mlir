cuda_tile.module @cuda_tile_module {
  entry @branch_a_only(%in_a: tile<ptr<f32>>,
                        %in_b: tile<ptr<f32>>,
                        %out: tile<ptr<f32>>) {
    %c0_i32 = constant <i32: 0> : tile<i32>
    %tview_a = make_tensor_view %in_a, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
    %pview_a = make_partition_view %tview_a : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
    %A, %tok_a = load_view_tko weak %pview_a[%c0_i32, %c0_i32] : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> tile<128x128xf32>, token

    %tview_b = make_tensor_view %in_b, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
    %pview_b = make_partition_view %tview_b : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
    %B, %tok_b = load_view_tko weak %pview_b[%c0_i32, %c0_i32] : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> tile<128x128xf32>, token

    %zero_mat = constant <f32: 0.000000e+00> : tile<128x128xf32>
    %C = mmaf %A, %B, %zero_mat : tile<128x128xf32>, tile<128x128xf32>, tile<128x128xf32>

    %tview_out_a = make_tensor_view %out, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
    %pview_out_a = make_partition_view %tview_out_a : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
    %tok_out_a = store_view_tko weak %C, %pview_out_a[%c0_i32, %c0_i32] : tile<128x128xf32>, partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> token
    return
  }
}
