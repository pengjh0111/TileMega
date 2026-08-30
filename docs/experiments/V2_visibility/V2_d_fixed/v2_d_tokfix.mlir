// V1-d: all three bugs fixed AND the contention pattern changed to match a real
// event-driven megakernel.
//
// V1-ctrl/a/b all have every consumer spinning on ONE shared flag address --
// the worst imaginable contention. A per-tile event tensor instead gives each
// event a handful of waiters, spread over many addresses. This variant models
// that:
//   * 4 producers (bx < 4). Producer p writes data region p (64 chunks) and
//     then releases the flag of every consumer assigned to it.
//   * Consumer bx waits on flag[bx*32] alone -- fan-out 1 per event.
//     The *32 stride pads each flag to its own 128B cache line (skeleton 6.5).
//   * Consumer bx reduces region (bx & 3) -> 64 * 1024 = 65536.0
cuda_tile.module @cuda_tile_module {
  entry @v2_d_tokfix(%data: tile<ptr<f32>>,
              %flag: tile<ptr<i32>>,
              %out:  tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %ngrid, %ny, %nz = get_num_tile_blocks : tile<i32>
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %c3 = constant <i32: 3> : tile<i32>
    %c4 = constant <i32: 4> : tile<i32>
    %c32 = constant <i32: 32> : tile<i32>
    %c64 = constant <i32: 64> : tile<i32>

    %is_prod = cmpi less_than %bx, %c4, signed : tile<i32> -> tile<i1>
    if %is_prod {
      %dv = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %dp = make_partition_view %dv : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %one_1f = constant <f32: 1.000000e+00> : tile<1xf32>
      %ones = broadcast %one_1f : tile<1xf32> -> tile<1024xf32>
      %init = make_token : token

      // Producer p writes chunks [p*64, p*64+64).
      %lo = muli %bx, %c64 : tile<i32>
      %hi = addi %lo, %c64 : tile<i32>
      %chain = for %c in (%lo to %hi, step %c1) : tile<i32>
                   iter_values(%tok = %init) -> (token) {
        %t = store_view_tko weak %ones, %dp[%c] token=%tok : tile<1024xf32>, partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> token
        continue %t : token
      }

      // Then release the flag of every consumer j with (j mod 4) == p.
      // Each store is a RELEASE token-ordered after the whole data chain
      // (rule 6.8), so each consumer's acquire sees this producer's region.
      %fv = make_tensor_view %flag, shape = [16384], strides = [1] : tensor_view<16384xi32, strides=[1]>
      %fp = make_partition_view %fv : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>
      %one_1i = constant <i32: 1> : tile<1xi32>
      %jlo = addi %bx, %c4 : tile<i32>
      %ftok = for %j in (%jlo to %ngrid, step %c4) : tile<i32>
                  iter_values(%tk = %chain) -> (token) {
        %fidx = muli %j, %c32 : tile<i32>
        %ft = store_view_tko release device %one_1i, %fp[%fidx] token=%tk : tile<1xi32>, partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> token
        continue %ft : token
      }
      yield
    } else {
      %fv2 = make_tensor_view %flag, shape = [16384], strides = [1] : tensor_view<16384xi32, strides=[1]>
      %fp2 = make_partition_view %fv2 : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>
      %myflag = muli %bx, %c32 : tile<i32>
      %it = make_token : token

      // Spin on this block's OWN flag: fan-out 1, no shared hot address.
      %lt = loop iter_values(%tok = %it) : token -> token {
        %v, %t2 = load_view_tko relaxed device %fp2[%myflag] token=%tok : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
        %vs = reshape %v : tile<1xi32> -> tile<i32>
        %rd = cmpi equal %vs, %c1, signed : tile<i32> -> tile<i1>
        if %rd { break %t2 : token }
        continue %t2 : token
      }
      %a, %at = load_view_tko acquire device %fp2[%myflag] token=%lt : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token

      %dv2 = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %dp2 = make_partition_view %dv2 : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %region = andi %bx, %c3 : tile<i32>
      %rlo = muli %region, %c64 : tile<i32>
      %rhi = addi %rlo, %c64 : tile<i32>
      %zero = constant <f32: 0.000000e+00> : tile<f32>
      %sum = for %c2 in (%rlo to %rhi, step %c1) : tile<i32>
                 iter_values(%acc = %zero) -> (tile<f32>) {
        %d, %dt = load_view_tko relaxed device %dp2[%c2] token=%at : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
        %cs = reduce %d dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
          (%e: tile<f32>, %ra: tile<f32>) { %z = addf %e, %ra : tile<f32>
            yield %z : tile<f32> }
        %na = addf %acc, %cs : tile<f32>
        continue %na : tile<f32>
      }

      %r1 = reshape %sum : tile<f32> -> tile<1xf32>
      %ov = make_tensor_view %out, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
      %op = make_partition_view %ov : partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>
      %ot = store_view_tko weak %r1, %op[%bx] : tile<1xf32>, partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
      yield
    }
    return
  }
}
