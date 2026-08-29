// E2: heterogeneous tile dispatch via runtime scalar task_type.
// task_type == 0  -> branch A: tile<128x128xf32> mmaf (matmul-like workload)
// task_type != 0  -> branch B: tile<256xf32> reduce (sum reduction workload)
// Both branches operate purely inside the `if`, without yielding any SSA
// value out of the if (per E1 Q6 finding: IfOp only requires matching types
// across branches when values are yielded out).
cuda_tile.module @cuda_tile_module {
  entry @heterogeneous_dispatch(%task_type: tile<i32>,
                                 %in_a: tile<ptr<f32>>,
                                 %in_b: tile<ptr<f32>>,
                                 %out: tile<ptr<f32>>) {
    %c0_i32 = constant <i32: 0> : tile<i32>
    %cond = cmpi equal %task_type, %c0_i32, signed : tile<i32> -> tile<i1>
    if %cond {
      // ---- Branch A: 128x128 matmul-like mmaf ----
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
      yield
    } else {
      // ---- Branch B: 256-wide sum reduction ----
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
      yield
    }
    return
  }
}
