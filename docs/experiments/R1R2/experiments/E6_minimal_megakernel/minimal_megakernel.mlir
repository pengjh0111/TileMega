// E6: minimal "megakernel" — one persistent kernel launch containing three
// concurrently-resident tile blocks with heterogeneous roles and one real
// cross-block dependency edge, combining:
//   - E2's per-block heterogeneous dispatch (branch on get_tile_block_id)
//   - E4's token-chained spin-wait fix (the only pattern proven not to be
//     silently eliminated by the compiler)
//   - a genuinely independent third task that does NOT depend on the
//     producer/consumer pair, to test whether it can make progress without
//     being blocked by the producer/consumer synchronization
//
// Roles (by tile-block x id):
//   block 0 (producer):   C[0..262143] = A + B, chunked (256 x 1024),
//                          bumping an atomic `progress` counter once per
//                          chunk (token-chained across loop iterations, per
//                          the E4 lesson that un-chained tko ops can be
//                          silently eliminated), then release-xchg %flag=1.
//   block 1 (consumer):   token-chained spin-wait on %flag (E4's proven-working
//                          pattern), then D[0..262143] = C * 2.0, chunked.
//   block 2 (independent): does NOT wait on %flag at all. It takes one
//                          "snapshot" read of the producer's progress counter
//                          (a single, non-looped weak load -- not subject to
//                          the loop-elimination bug since there's no loop),
//                          stores that snapshot to %block2_sample, then does
//                          its own unrelated work E[0..1023] = X * 3.0.
//                          If %block2_sample ends up < 256 (the producer's
//                          final per-chunk counter value), that is evidence
//                          block 2 actually ran concurrently with (not after)
//                          the producer's full completion, i.e. real overlap
//                          within a single kernel launch.
cuda_tile.module @cuda_tile_module {
  entry @minimal_megakernel(%A: tile<ptr<f32>>,
                             %B: tile<ptr<f32>>,
                             %C: tile<ptr<f32>>,
                             %D: tile<ptr<f32>>,
                             %X: tile<ptr<f32>>,
                             %E: tile<ptr<f32>>,
                             %flag: tile<ptr<i32>>,
                             %progress: tile<ptr<i32>>,
                             %block2_sample: tile<ptr<i32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %c0_i32 = constant <i32: 0> : tile<i32>
    %c1_i32 = constant <i32: 1> : tile<i32>
    %is_producer = cmpi equal %bx, %c0_i32, signed : tile<i32> -> tile<i1>
    if %is_producer {
      // ---- block 0: producer, C = A + B, bump progress each chunk, then signal flag ----
      %a_view = make_tensor_view %A, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %a_pview = make_partition_view %a_view : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %b_view = make_tensor_view %B, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %b_pview = make_partition_view %b_view : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %c_view = make_tensor_view %C, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %c_pview = make_partition_view %c_view : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %progress_1 = reshape %progress : tile<ptr<i32>> -> tile<1xptr<i32>>
      %one_1 = constant <i32: 1> : tile<1xi32>
      %num_chunks = constant <i32: 256> : tile<i32>
      %init_tok_p = make_token : token
      %final_tok_p = loop iter_values(%tok = %init_tok_p, %chunk = %c0_i32) : token, tile<i32> -> token {
        %a_tile, %tok_a = load_view_tko weak %a_pview[%chunk] token=%tok : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
        %b_tile, %tok_b = load_view_tko weak %b_pview[%chunk] token=%tok_a : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
        %c_tile = addf %a_tile, %b_tile : tile<1024xf32>
        %tok_c = store_view_tko weak %c_tile, %c_pview[%chunk] token=%tok_b : tile<1024xf32>, partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> token
        %old_progress, %tok_prog = atomic_rmw_tko release device %progress_1, add, %one_1 token=%tok_c : tile<1xptr<i32>>, tile<1xi32> -> tile<1xi32>, token
        %next_chunk = addi %chunk, %c1_i32 : tile<i32>
        %done = cmpi equal %next_chunk, %num_chunks, signed : tile<i32> -> tile<i1>
        if %done {
          break %tok_prog : token
        }
        continue %tok_prog, %next_chunk : token, tile<i32>
      }
      %flag_1 = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
      %old_flag, %tok_flag = atomic_rmw_tko release device %flag_1, xchg, %one_1 token=%final_tok_p : tile<1xptr<i32>>, tile<1xi32> -> tile<1xi32>, token
      yield
    } else {
      %is_consumer = cmpi equal %bx, %c1_i32, signed : tile<i32> -> tile<i1>
      if %is_consumer {
        // ---- block 1: consumer, token-chained spin-wait then D = C * 2.0 ----
        %flag_1c = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
        %init_tok_c = make_token : token
        %final_tok_c = loop iter_values(%tok = %init_tok_c) : token -> token {
          %val, %ltok = load_ptr_tko acquire device %flag_1c token=%tok : tile<1xptr<i32>> -> tile<1xi32>, token
          %val_s = reshape %val : tile<1xi32> -> tile<i32>
          %c1_check = constant <i32: 1> : tile<i32>
          %is_set = cmpi equal %val_s, %c1_check, signed : tile<i32> -> tile<i1>
          if %is_set {
            break %ltok : token
          }
          continue %ltok : token
        }
        %c_view2 = make_tensor_view %C, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
        %c_pview2 = make_partition_view %c_view2 : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
        %d_view = make_tensor_view %D, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
        %d_pview = make_partition_view %d_view : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
        %num_chunks2 = constant <i32: 256> : tile<i32>
        %two_f = constant <f32: 2.000000e+00> : tile<1024xf32>
        %final_tok_d = loop iter_values(%tok2 = %final_tok_c, %chunk2 = %c0_i32) : token, tile<i32> -> token {
          %c_tile, %tok_ld = load_view_tko acquire device %c_pview2[%chunk2] token=%tok2 : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
          %d_tile = mulf %c_tile, %two_f : tile<1024xf32>
          %tok_st = store_view_tko weak %d_tile, %d_pview[%chunk2] token=%tok_ld : tile<1024xf32>, partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> token
          %next_chunk2 = addi %chunk2, %c1_i32 : tile<i32>
          %done2 = cmpi equal %next_chunk2, %num_chunks2, signed : tile<i32> -> tile<i1>
          if %done2 {
            break %tok_st : token
          }
          continue %tok_st, %next_chunk2 : token, tile<i32>
        }
        yield
      } else {
        // ---- block 2 (and any further blocks): independent task, NOT waiting on %flag ----
        %progress_1s = reshape %progress : tile<ptr<i32>> -> tile<1xptr<i32>>
        %sample, %tok_sample = load_ptr_tko weak %progress_1s : tile<1xptr<i32>> -> tile<1xi32>, token
        %sample_out = reshape %block2_sample : tile<ptr<i32>> -> tile<1xptr<i32>>
        %tok_store_sample = store_ptr_tko weak %sample_out, %sample token=%tok_sample : tile<1xptr<i32>>, tile<1xi32> -> token
        %x_view = make_tensor_view %X, shape = [1024], strides = [1] : tensor_view<1024xf32, strides=[1]>
        %x_pview = make_partition_view %x_view : partition_view<tile=(1024), tensor_view<1024xf32, strides=[1]>>
        %e_view = make_tensor_view %E, shape = [1024], strides = [1] : tensor_view<1024xf32, strides=[1]>
        %e_pview = make_partition_view %e_view : partition_view<tile=(1024), tensor_view<1024xf32, strides=[1]>>
        %x_tile, %tok_x = load_view_tko weak %x_pview[%c0_i32] : partition_view<tile=(1024), tensor_view<1024xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
        %three_f = constant <f32: 3.000000e+00> : tile<1024xf32>
        %e_tile = mulf %x_tile, %three_f : tile<1024xf32>
        %tok_e = store_view_tko weak %e_tile, %e_pview[%c0_i32] : tile<1024xf32>, partition_view<tile=(1024), tensor_view<1024xf32, strides=[1]>>, tile<i32> -> token
        yield
      }
    }
    return
  }
}
