cuda_tile.module @cuda_tile_module {
  entry @branch_b_only(%in_a: tile<ptr<f32>>,
                        %in_b: tile<ptr<f32>>,
                        %out: tile<ptr<f32>>) {
    %c0_i32 = constant <i32: 0> : tile<i32>
    %tview_v = make_tensor_view %in_a, shape = [256], strides = [1] : tensor_view<256xf32, strides=[1]>
    %pview_v = make_partition_view %tview_v : partition_view<tile=(256), tensor_view<256xf32, strides=[1]>>
    %vec, %tok_v = load_view_tko weak %pview_v[%c0_i32] : partition_view<tile=(256), tensor_view<256xf32, strides=[1]>>, tile<i32> -> tile<256xf32>, token

    %sum = reduce %vec dim=0 identities=[0.000000e+00 : f32] : tile<256xf32> -> tile<f32>
      (%elem: tile<f32>, %acc: tile<f32>) {
        %add = addf %elem, %acc : tile<f32>
        yield %add : tile<f32>
      }

    %sum_1d = reshape %sum : tile<f32> -> tile<1xf32>
    %tview_out_b = make_tensor_view %out, shape = [1], strides = [1] : tensor_view<1xf32, strides=[1]>
    %pview_out_b = make_partition_view %tview_out_b : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>
    %tok_out_b = store_view_tko weak %sum_1d, %pview_out_b[%c0_i32] : tile<1xf32>, partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>, tile<i32> -> token
    return
  }
}
