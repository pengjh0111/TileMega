
cuda_tile.module @cuda_tile_module {
  entry @v2_y_dist_shared(%data: tile<ptr<f32>>, %flag: tile<ptr<i32>>, %out: tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %ngrid, %ny, %nz = get_num_tile_blocks : tile<i32>
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %c32 = constant <i32: 32> : tile<i32>

    %dv = make_tensor_view %data, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
    %dp = make_partition_view %dv : partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>
    %fv = make_tensor_view %flag, shape = [16384], strides = [1] : tensor_view<16384xi32, strides=[1]>
    %fp = make_partition_view %fv : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>
    %one_1i = constant <i32: 1> : tile<1xi32>

    %is_prod = cmpi equal %bx, %c0, signed : tile<i32> -> tile<i1>
    if %is_prod {
      %one_1f = constant <f32: 1.000000e+00> : tile<1xf32>
      %ones = broadcast %one_1f : tile<1xf32> -> tile<1024xf32>
      %init = make_token : token
      %chain = for %c in (%c0 to %ngrid, step %c1) : tile<i32>
                   iter_values(%tok = %init) -> (token) {
        %t = store_view_tko relaxed device %ones, %dp[%c] token=%tok : tile<1024xf32>, partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
        continue %t : token
      }
      %ftok = for %j in (%c1 to %ngrid, step %c1) : tile<i32>
                  iter_values(%tk = %chain) -> (token) {
        %fidx = muli %j, %c32 : tile<i32>
        %ft = store_view_tko release device %one_1i, %fp[%fidx] token=%tk : tile<1xi32>, partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> token
        continue %ft : token
      }
      yield
    } else {
      %myslot = muli %bx, %c32 : tile<i32>
      %it = make_token : token
      %lt = loop iter_values(%tok = %it) : token -> token {
        %v, %t2 = load_view_tko relaxed device %fp[%myslot] token=%tok : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
        %vs = reshape %v : tile<1xi32> -> tile<i32>
        %rd = cmpi equal %vs, %c1, signed : tile<i32> -> tile<i1>
        if %rd { break %t2 : token }
        continue %t2 : token
      }
      %a, %at = load_view_tko acquire device %fp[%myslot] token=%lt : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %d, %dt = load_view_tko relaxed device %dp[%c0] token=%at : partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      %s = reduce %d dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
        (%e: tile<f32>, %ra: tile<f32>) { %z = addf %e, %ra : tile<f32>
          yield %z : tile<f32> }
      %r1 = reshape %s : tile<f32> -> tile<1xf32>
      %ov = make_tensor_view %out, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
      %op = make_partition_view %ov : partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>
      %ot = store_view_tko relaxed device %r1, %op[%bx] token=%dt : tile<1xf32>, partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
      yield
    }
    return
  }
}
