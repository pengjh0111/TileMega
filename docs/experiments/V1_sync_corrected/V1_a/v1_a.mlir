// V1-a: fixes Bug A (release token-ordered after the data stores) and Bug C
// (each block writes its own output slot). The consumer is ELEMENTWISE -- no
// reduction -- to test whether the synchronisation mechanism itself works at
// large grid sizes.
//
// Data contract:
//   data[0..262143] = 1.0f   (host pre-fills with -1.0f poison)
//   flag[0]                  release/acquire flag
//   out[bx*1024 .. +1023]    = 2.0f, written only by consumer block bx
cuda_tile.module @cuda_tile_module {
  entry @v1_a(%data: tile<ptr<f32>>,
              %flag: tile<ptr<i32>>,
              %out:  tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %c256 = constant <i32: 256> : tile<i32>
    %c255 = constant <i32: 255> : tile<i32>

    %is_prod = cmpi equal %bx, %c0, signed : tile<i32> -> tile<i1>
    if %is_prod {
      %dv = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %dp = make_partition_view %dv : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %one_1f = constant <f32: 1.000000e+00> : tile<1xf32>
      %ones = broadcast %one_1f : tile<1xf32> -> tile<1024xf32>
      %init = make_token : token

      // Rule 6.8: carry the store token out of the loop so the release below
      // can be token-ordered after every store. Program order alone provides
      // no ordering (spec 7.5).
      %chain = for %c in (%c0 to %c256, step %c1) : tile<i32>
                   iter_values(%tok = %init) -> (token) {
        %t = store_view_tko weak %ones, %dp[%c] token=%tok : tile<1024xf32>, partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> token
        continue %t : token
      }

      %flag_1 = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
      %one_1 = constant <i32: 1> : tile<1xi32>
      %old, %atok = atomic_rmw_tko release device %flag_1, xchg, %one_1 token=%chain : tile<1xptr<i32>>, tile<1xi32> -> tile<1xi32>, token
      yield
    } else {
      %flag_1c = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
      %init_tok = make_token : token

      // Rule 6.9: poll with relaxed (no cache invalidation per iteration),
      // then do ONE acquire after leaving the loop to establish happens-before.
      %loop_tok = loop iter_values(%tok = %init_tok) : token -> token {
        %val, %ltok = load_ptr_tko relaxed device %flag_1c token=%tok : tile<1xptr<i32>> -> tile<1xi32>, token
        %val_s = reshape %val : tile<1xi32> -> tile<i32>
        %is_set = cmpi equal %val_s, %c1, signed : tile<i32> -> tile<i1>
        if %is_set {
          break %ltok : token
        }
        continue %ltok : token
      }
      %acq, %acq_tok = load_ptr_tko acquire device %flag_1c token=%loop_tok : tile<1xptr<i32>> -> tile<1xi32>, token

      // Data load is WEAK, ordered after the acquire purely by token (spec
      // 7.12.3). An `acquire device` here would expand to 1024 acquires.
      %dv2 = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %dp2 = make_partition_view %dv2 : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %chunk = andi %bx, %c255 : tile<i32>
      %d, %ltok2 = load_view_tko weak %dp2[%chunk] token=%acq_tok : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token

      %two_1f = constant <f32: 2.000000e+00> : tile<1xf32>
      %twos = broadcast %two_1f : tile<1xf32> -> tile<1024xf32>
      %res = mulf %d, %twos : tile<1024xf32>

      // Rule 6.10: per-block output slot; no two blocks write the same address.
      %ov = make_tensor_view %out, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
      %op = make_partition_view %ov : partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>
      %ot = store_view_tko weak %res, %op[%bx] token=%ltok2 : tile<1024xf32>, partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
      yield
    }
    return
  }
}
