// V2-f1: V1-min with the tile size and the reduction removed.
//
// Everything else is byte-for-byte the V1-min shape: one shared flag, every
// consumer reads the SAME data location, producer writes then releases.
// The only change is that the consumer loads a single element and stores it
// straight out, so the observed value IS the value that came back from memory
// rather than a sum that several different loss patterns could produce.
//
//   expect out[bx] == 1.0
//   poison  == -1.0   (so a stale read is directly visible as -1)
cuda_tile.module @cuda_tile_module {
  entry @v2_f1(%data: tile<ptr<f32>>, %flag: tile<ptr<i32>>, %out: tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %is_prod = cmpi equal %bx, %c0, signed : tile<i32> -> tile<i1>
    if %is_prod {
      %dv = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %dp = make_partition_view %dv : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %one_1f = constant <f32: 1.000000e+00> : tile<1xf32>
      %ones = broadcast %one_1f : tile<1xf32> -> tile<1024xf32>
      %st = store_view_tko relaxed device %ones, %dp[%c0] : tile<1024xf32>, partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> token
      %flag_1 = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
      %one_1 = constant <i32: 1> : tile<1xi32>
      %old, %atok = atomic_rmw_tko release device %flag_1, xchg, %one_1 token=%st : tile<1xptr<i32>>, tile<1xi32> -> tile<1xi32>, token
      yield
    } else {
      %flag_1c = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
      %it = make_token : token
      %lt = loop iter_values(%tok = %it) : token -> token {
        %v, %t2 = load_ptr_tko relaxed device %flag_1c token=%tok : tile<1xptr<i32>> -> tile<1xi32>, token
        %vs = reshape %v : tile<1xi32> -> tile<i32>
        %rd = cmpi equal %vs, %c1, signed : tile<i32> -> tile<i1>
        if %rd { break %t2 : token }
        continue %t2 : token
      }
      %a, %at = load_ptr_tko acquire device %flag_1c token=%lt : tile<1xptr<i32>> -> tile<1xi32>, token
      // SINGLE element: one LDG, no multi-instruction expansion, no reduce.
      %dv2 = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %dp2 = make_partition_view %dv2 : partition_view<tile=(1), tensor_view<262144xf32, strides=[1]>>
      %d, %dt = load_view_tko relaxed device %dp2[%c0] token=%at : partition_view<tile=(1), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> tile<1xf32>, token
      %ov = make_tensor_view %out, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
      %op = make_partition_view %ov : partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>
      %ot = store_view_tko weak %d, %op[%bx] token=%dt : tile<1xf32>, partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
      yield
    }
    return
  }
}
