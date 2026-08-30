// V2-f-const: THE DISCRIMINATING TEST.
//
// Consumer spins on the flag exactly as V1-min does, then reduces a tile that
// is a pure register constant -- broadcast(1.0f) -- with NO data load at all.
// The sum cannot depend on anything the producer wrote, on cache state, or on
// cross-block visibility. It is 1024.0 by construction.
//
// If this still returns 768 / 512, the corruption V1 attributed to "cross-block
// data visibility" is not a visibility problem at all: it is the reduction (and
// therefore the intra-CTA BAR.SYNC it relies on) producing a wrong answer after
// a divergent spin-loop exit.
cuda_tile.module @cuda_tile_module {
  entry @v2_f_const(%data: tile<ptr<f32>>, %flag: tile<ptr<i32>>, %out: tile<ptr<f32>>) {
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
      // NO data load. A register constant, reduced across the whole tile.
      %k_1f = constant <f32: 1.000000e+00> : tile<1xf32>
      %k = broadcast %k_1f : tile<1xf32> -> tile<1024xf32>
      %s = reduce %k dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
        (%e: tile<f32>, %ra: tile<f32>) { %z = addf %e, %ra : tile<f32>
          yield %z : tile<f32> }
      %r1 = reshape %s : tile<f32> -> tile<1xf32>
      %ov = make_tensor_view %out, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
      %op = make_partition_view %ov : partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>
      %ot = store_view_tko weak %r1, %op[%bx] token=%at : tile<1xf32>, partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
      yield
    }
    return
  }
}
