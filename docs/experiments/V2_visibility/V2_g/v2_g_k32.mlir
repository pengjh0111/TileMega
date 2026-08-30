
cuda_tile.module @cuda_tile_module {
  entry @v2_g_k32(%data: tile<ptr<f32>>, %flag: tile<ptr<i32>>, %out: tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %cP = constant <i32: 1> : tile<i32>
    %cK = constant <i32: 32> : tile<i32>
    %dv = make_tensor_view %data, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
    %dp = make_partition_view %dv : partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>
    %flag_1 = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
    %one_1i = constant <i32: 1> : tile<1xi32>
    %is_prod = cmpi equal %bx, %c0, signed : tile<i32> -> tile<i1>
    if %is_prod {
      %one_1f = constant <f32: 1.000000e+00> : tile<1xf32>
      %ones = broadcast %one_1f : tile<1xf32> -> tile<1024xf32>
      %init = make_token : token
      %chain = for %c in (%c0 to %cP, step %c1) : tile<i32>
                   iter_values(%tok = %init) -> (token) {
        %t = store_view_tko relaxed device %ones, %dp[%c] token=%tok : tile<1024xf32>, partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
        continue %t : token
      }
      %old, %atok = atomic_rmw_tko release device %flag_1, xchg, %one_1i token=%chain : tile<1xptr<i32>>, tile<1xi32> -> tile<1xi32>, token
      yield
    } else {
      %part = cmpi less_than_or_equal %bx, %cK, signed : tile<i32> -> tile<i1>
      if %part {
        %it = make_token : token
        %lt = loop iter_values(%tok = %it) : token -> token {
          %v, %t2 = load_ptr_tko relaxed device %flag_1 token=%tok : tile<1xptr<i32>> -> tile<1xi32>, token
          %vs = reshape %v : tile<1xi32> -> tile<i32>
          %rd = cmpi equal %vs, %c1, signed : tile<i32> -> tile<i1>
          if %rd { break %t2 : token }
          continue %t2 : token
        }
        %a, %at = load_ptr_tko acquire device %flag_1 token=%lt : tile<1xptr<i32>> -> tile<1xi32>, token
        %zero = constant <f32: 0.000000e+00> : tile<f32>
        %sum = for %c2 in (%c0 to %cP, step %c1) : tile<i32>
                   iter_values(%acc = %zero) -> (tile<f32>) {
          %d, %dt = load_view_tko relaxed device %dp[%c2] token=%at : partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
          %cs = reduce %d dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
            (%e: tile<f32>, %ra: tile<f32>) { %z = addf %e, %ra : tile<f32>
              yield %z : tile<f32> }
          %na = addf %acc, %cs : tile<f32>
          continue %na : tile<f32>
        }
        %r1 = reshape %sum : tile<f32> -> tile<1xf32>
        %ov = make_tensor_view %out, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
        %op = make_partition_view %ov : partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>
        %ot = store_view_tko relaxed device %r1, %op[%bx] token=%at : tile<1xf32>, partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
        yield
      } else { yield }
      yield
    }
    return
  }
}
