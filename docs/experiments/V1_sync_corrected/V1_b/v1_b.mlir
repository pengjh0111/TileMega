// V1-b: fixes Bug A, Bug B and Bug C. The consumer KEEPS the 256-iteration
// reduction, so the only difference from V1-a is the reduce itself. Comparing
// the two isolates whether the reduce is implicated in the hang, and comparing
// against V1-ctrl isolates the cost of Bug B (`acquire device` on every element
// of a tile<1024xf32>, i.e. 1024 acquires per instruction, spec 7.1).
//
// Data contract:
//   data[0..262143] = 1.0f   (host pre-fills with -1.0f poison)
//   out[bx]         = 262144.0f, written only by consumer block bx
cuda_tile.module @cuda_tile_module {
  entry @v1_b(%data: tile<ptr<f32>>,
              %flag: tile<ptr<i32>>,
              %out:  tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %c256 = constant <i32: 256> : tile<i32>

    %is_prod = cmpi equal %bx, %c0, signed : tile<i32> -> tile<i1>
    if %is_prod {
      %dv = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %dp = make_partition_view %dv : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %one_1f = constant <f32: 1.000000e+00> : tile<1xf32>
      %ones = broadcast %one_1f : tile<1xf32> -> tile<1024xf32>
      %init = make_token : token
      // Rule 6.8: the release must be token-ordered after every store.
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
      %loop_tok = loop iter_values(%tok = %init_tok) : token -> token {
        %val, %ltok = load_ptr_tko relaxed device %flag_1c token=%tok : tile<1xptr<i32>> -> tile<1xi32>, token
        %val_s = reshape %val : tile<1xi32> -> tile<i32>
        %is_set = cmpi equal %val_s, %c1, signed : tile<i32> -> tile<i1>
        if %is_set {
          break %ltok : token
        }
        continue %ltok : token
      }
      // Rule 6.9: exactly ONE acquire, on the scalar flag.
      %acq, %acq_tok = load_ptr_tko acquire device %flag_1c token=%loop_tok : tile<1xptr<i32>> -> tile<1xi32>, token

      %dv2 = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %dp2 = make_partition_view %dv2 : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %zero = constant <f32: 0.000000e+00> : tile<f32>
      // The chunk loads are WEAK and each is token-ordered after the single
      // acquire. They need no mutual ordering (spec 7.12.3), so the token is
      // not chained between iterations.
      %sum = for %c2 in (%c0 to %c256, step %c1) : tile<i32>
                 iter_values(%acc = %zero) -> (tile<f32>) {
        %d, %lt = load_view_tko weak %dp2[%c2] token=%acq_tok : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
        %cs = reduce %d dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
          (%e: tile<f32>, %ra: tile<f32>) {
            %s = addf %e, %ra : tile<f32>
            yield %s : tile<f32>
          }
        %na = addf %acc, %cs : tile<f32>
        continue %na : tile<f32>
      }

      // Rule 6.10: per-block output slot.
      %res = reshape %sum : tile<f32> -> tile<1xf32>
      %ov = make_tensor_view %out, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
      %op = make_partition_view %ov : partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>
      %ot = store_view_tko weak %res, %op[%bx] : tile<1xf32>, partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
      yield
    }
    return
  }
}
