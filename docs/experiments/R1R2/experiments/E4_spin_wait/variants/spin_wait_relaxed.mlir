// E4 (heart of megakernel): single entry, 2 tile blocks by role.
// Block 0 (tile_block_id.x == 0) = producer: writes a 1MB (262144xf32) pattern
//   in 256 chunks of 1024 elements via weak store_view_tko, then sets a flag
//   via a RELEASE-scoped atomic_rmw_tko xchg (device scope).
// Block 1 (tile_block_id.x == 1) = consumer: spin-waits on the flag via an
//   unbounded LoopOp doing an ACQUIRE-scoped load_ptr_tko + if/break, then
//   re-reads the entire 1MB buffer and computes a checksum (sum of all
//   elements) which is written to `checksum_out` for host-side comparison
//   against a CPU-computed reference sum.
cuda_tile.module @cuda_tile_module {
  entry @spin_wait_relaxed(%data: tile<ptr<f32>>,
                         %flag: tile<ptr<i32>>,
                         %checksum_out: tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %c0_i32 = constant <i32: 0> : tile<i32>
    %is_producer = cmpi equal %bx, %c0_i32, signed : tile<i32> -> tile<i1>
    if %is_producer {
      // ---- Producer ----
      %data_view = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %data_pview = make_partition_view %data_view : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %num_chunks = constant <i32: 256> : tile<i32>
      %c1_i32 = constant <i32: 1> : tile<i32>
      %c1024_i32 = constant <i32: 1024> : tile<i32>
      for %chunk in (%c0_i32 to %num_chunks, step %c1_i32) : tile<i32> {
        %base = muli %chunk, %c1024_i32 : tile<i32>
        %local_iota = iota : tile<1024xi32>
        %base_1d = reshape %base : tile<i32> -> tile<1xi32>
        %base_bcast = broadcast %base_1d : tile<1xi32> -> tile<1024xi32>
        %global_idx = addi %local_iota, %base_bcast : tile<1024xi32>
        %pattern_f = itof %global_idx signed : tile<1024xi32> -> tile<1024xf32>
        %tok_store = store_view_tko weak %pattern_f, %data_pview[%chunk] : tile<1024xf32>, partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> token
        continue
      }
      %flag_1 = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
      %one_1 = constant <i32: 1> : tile<1xi32>
      %old_flag, %atok = atomic_rmw_tko release device %flag_1, xchg, %one_1 : tile<1xptr<i32>>, tile<1xi32> -> tile<1xi32>, token
      yield
    } else {
      // ---- Consumer: spin-wait then verify ----
      %flag_1c = reshape %flag : tile<ptr<i32>> -> tile<1xptr<i32>>
      loop {
        %val, %ltok = load_ptr_tko relaxed device %flag_1c : tile<1xptr<i32>> -> tile<1xi32>, token
        %val_s = reshape %val : tile<1xi32> -> tile<i32>
        %c1_check = constant <i32: 1> : tile<i32>
        %is_set = cmpi equal %val_s, %c1_check, signed : tile<i32> -> tile<i1>
        if %is_set {
          break
        }
        continue
      }

      %data_view2 = make_tensor_view %data, shape = [262144], strides = [1] : tensor_view<262144xf32, strides=[1]>
      %data_pview2 = make_partition_view %data_view2 : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>
      %num_chunks2 = constant <i32: 256> : tile<i32>
      %c1_i32b = constant <i32: 1> : tile<i32>
      %zero_sum = constant <f32: 0.000000e+00> : tile<f32>
      %final_sum = for %chunk2 in (%c0_i32 to %num_chunks2, step %c1_i32b) : tile<i32>
                       iter_values(%acc = %zero_sum) -> (tile<f32>) {
        %chunk_data, %tok_load = load_view_tko acquire device %data_pview2[%chunk2] : partition_view<tile=(1024), tensor_view<262144xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
        %chunk_sum = reduce %chunk_data dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
          (%elem: tile<f32>, %redacc: tile<f32>) {
            %s = addf %elem, %redacc : tile<f32>
            yield %s : tile<f32>
          }
        %new_acc = addf %acc, %chunk_sum : tile<f32>
        continue %new_acc : tile<f32>
      }

      %checksum_1d = reshape %final_sum : tile<f32> -> tile<1xf32>
      %checksum_view = make_tensor_view %checksum_out, shape = [1], strides = [1] : tensor_view<1xf32, strides=[1]>
      %checksum_pview = make_partition_view %checksum_view : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>
      %tok_out = store_view_tko weak %checksum_1d, %checksum_pview[%c0_i32] : tile<1xf32>, partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>, tile<i32> -> token
      yield
    }
    return
  }
}
