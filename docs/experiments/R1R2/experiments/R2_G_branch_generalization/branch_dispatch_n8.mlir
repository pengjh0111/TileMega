cuda_tile.module @cuda_tile_module {
  entry @branch_dispatch_n8(%task_type: tile<i32>, %in_a: tile<ptr<f32>>, %in_b: tile<ptr<f32>>, %out: tile<ptr<f32>>) {
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
        %c2_i32 = constant <i32: 2> : tile<i32>
        %cond2 = cmpi equal %task_type, %c2_i32, signed : tile<i32> -> tile<i1>
        if %cond2 {

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
        } else {
          %c3_i32 = constant <i32: 3> : tile<i32>
          %cond3 = cmpi equal %task_type, %c3_i32, signed : tile<i32> -> tile<i1>
          if %cond3 {

      %tview_v3 = make_tensor_view %in_a, shape = [1024], strides = [1] : tensor_view<1024xf32, strides=[1]>
      %pview_v3 = make_partition_view %tview_v3 : partition_view<tile=(1024), tensor_view<1024xf32, strides=[1]>>
      %vec3, %tok_v3 = load_view_tko weak %pview_v3[%c0_i32] : partition_view<tile=(1024), tensor_view<1024xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      %sum3 = reduce %vec3 dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
        (%elem3: tile<f32>, %acc3: tile<f32>) {
          %add3 = addf %elem3, %acc3 : tile<f32>
          yield %add3 : tile<f32>
        }
      %sum_1d3 = reshape %sum3 : tile<f32> -> tile<1xf32>
      %tview_out3 = make_tensor_view %out, shape = [1], strides = [1] : tensor_view<1xf32, strides=[1]>
      %pview_out3 = make_partition_view %tview_out3 : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>
      %tok_out3 = store_view_tko weak %sum_1d3, %pview_out3[%c0_i32] : tile<1xf32>, partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>, tile<i32> -> token

            yield
          } else {
            %c4_i32 = constant <i32: 4> : tile<i32>
            %cond4 = cmpi equal %task_type, %c4_i32, signed : tile<i32> -> tile<i1>
            if %cond4 {

      %tview_a4 = make_tensor_view %in_a, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
      %pview_a4 = make_partition_view %tview_a4 : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
      %A4, %tok_a4 = load_view_tko weak %pview_a4[%c0_i32, %c0_i32] : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> tile<128x128xf32>, token
      %tview_b4 = make_tensor_view %in_b, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
      %pview_b4 = make_partition_view %tview_b4 : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
      %B4, %tok_b4 = load_view_tko weak %pview_b4[%c0_i32, %c0_i32] : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> tile<128x128xf32>, token
      %zero_mat4 = constant <f32: 0.000000e+00> : tile<128x128xf32>
      %C4 = mmaf %A4, %B4, %zero_mat4 : tile<128x128xf32>, tile<128x128xf32>, tile<128x128xf32>
      %tview_out4 = make_tensor_view %out, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
      %pview_out4 = make_partition_view %tview_out4 : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
      %tok_out4 = store_view_tko weak %C4, %pview_out4[%c0_i32, %c0_i32] : tile<128x128xf32>, partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> token

              yield
            } else {
              %c5_i32 = constant <i32: 5> : tile<i32>
              %cond5 = cmpi equal %task_type, %c5_i32, signed : tile<i32> -> tile<i1>
              if %cond5 {

      %tview_v5 = make_tensor_view %in_a, shape = [256], strides = [1] : tensor_view<256xf32, strides=[1]>
      %pview_v5 = make_partition_view %tview_v5 : partition_view<tile=(256), tensor_view<256xf32, strides=[1]>>
      %vec5, %tok_v5 = load_view_tko weak %pview_v5[%c0_i32] : partition_view<tile=(256), tensor_view<256xf32, strides=[1]>>, tile<i32> -> tile<256xf32>, token
      %sum5 = reduce %vec5 dim=0 identities=[0.000000e+00 : f32] : tile<256xf32> -> tile<f32>
        (%elem5: tile<f32>, %acc5: tile<f32>) {
          %add5 = addf %elem5, %acc5 : tile<f32>
          yield %add5 : tile<f32>
        }
      %sum_1d5 = reshape %sum5 : tile<f32> -> tile<1xf32>
      %tview_out5 = make_tensor_view %out, shape = [1], strides = [1] : tensor_view<1xf32, strides=[1]>
      %pview_out5 = make_partition_view %tview_out5 : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>
      %tok_out5 = store_view_tko weak %sum_1d5, %pview_out5[%c0_i32] : tile<1xf32>, partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>, tile<i32> -> token

                yield
              } else {
                %c6_i32 = constant <i32: 6> : tile<i32>
                %cond6 = cmpi equal %task_type, %c6_i32, signed : tile<i32> -> tile<i1>
                if %cond6 {

      %tview_a6 = make_tensor_view %in_a, shape = [64, 64], strides = [64, 1] : tensor_view<64x64xf32, strides=[64,1]>
      %pview_a6 = make_partition_view %tview_a6 : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>
      %A6, %tok_a6 = load_view_tko weak %pview_a6[%c0_i32, %c0_i32] : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>, tile<i32> -> tile<64x64xf32>, token
      %tview_b6 = make_tensor_view %in_b, shape = [64, 64], strides = [64, 1] : tensor_view<64x64xf32, strides=[64,1]>
      %pview_b6 = make_partition_view %tview_b6 : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>
      %B6, %tok_b6 = load_view_tko weak %pview_b6[%c0_i32, %c0_i32] : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>, tile<i32> -> tile<64x64xf32>, token
      %zero_mat6 = constant <f32: 0.000000e+00> : tile<64x64xf32>
      %C6 = mmaf %A6, %B6, %zero_mat6 : tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>
      %tview_out6 = make_tensor_view %out, shape = [64, 64], strides = [64, 1] : tensor_view<64x64xf32, strides=[64,1]>
      %pview_out6 = make_partition_view %tview_out6 : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>
      %tok_out6 = store_view_tko weak %C6, %pview_out6[%c0_i32, %c0_i32] : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>, tile<i32> -> token

                  yield
                } else {

      %tview_v7 = make_tensor_view %in_a, shape = [1024], strides = [1] : tensor_view<1024xf32, strides=[1]>
      %pview_v7 = make_partition_view %tview_v7 : partition_view<tile=(1024), tensor_view<1024xf32, strides=[1]>>
      %vec7, %tok_v7 = load_view_tko weak %pview_v7[%c0_i32] : partition_view<tile=(1024), tensor_view<1024xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      %sum7 = reduce %vec7 dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
        (%elem7: tile<f32>, %acc7: tile<f32>) {
          %add7 = addf %elem7, %acc7 : tile<f32>
          yield %add7 : tile<f32>
        }
      %sum_1d7 = reshape %sum7 : tile<f32> -> tile<1xf32>
      %tview_out7 = make_tensor_view %out, shape = [1], strides = [1] : tensor_view<1xf32, strides=[1]>
      %pview_out7 = make_partition_view %tview_out7 : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>
      %tok_out7 = store_view_tko weak %sum_1d7, %pview_out7[%c0_i32] : tile<1xf32>, partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>, tile<i32> -> token

                  yield
                }
              }
            }
          }
        }
      }
    }
    return
  }
}
