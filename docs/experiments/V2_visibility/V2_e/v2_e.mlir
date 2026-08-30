// V2-e: 事件与数据 *都* 分布式 —— 真实 megakernel 任务图的形态。
//
// 每个 block 既是生产者又是消费者，构成一条链：
//   block bx  生产 chunk (bx+1)，然后 release flag[(bx+1)*32]
//   block bx  等待 flag[bx*32]，读 *自己独占的* chunk bx，归约，写 out[bx]
// 于是：
//   * 每个 flag slot 恰好 1 个写者 + 1 个读者（fan-out 1，无热点地址）
//   * 每个 chunk 恰好 1 个写者 + 1 个读者（数据完全不共享）
//   * flag slot 间隔 32 个 int = 128B，各占独立 cache line（骨架 6.5）
//   * 所有数据 load/store 一律 relaxed device，绝不用 weak（规范 7.2）
// 依赖方向永远是 bx -> bx+1（低号 block 先就绪），因此不需要 cooperative launch。
// chunk 0 与 flag[0] 由 host 预置，block 0 因此不会空等。
cuda_tile.module @cuda_tile_module {
  entry @v2_e(%data: tile<ptr<f32>>, %flag: tile<ptr<i32>>, %out: tile<ptr<f32>>) {
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %c32 = constant <i32: 32> : tile<i32>

    %dv = make_tensor_view %data, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
    %dp = make_partition_view %dv : partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>
    %fv = make_tensor_view %flag, shape = [16384], strides = [1] : tensor_view<16384xi32, strides=[1]>
    %fp = make_partition_view %fv : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>

    // ---- 生产者角色：写 chunk bx+1，再 release 下游的 flag ----
    %next = addi %bx, %c1 : tile<i32>
    %one_1f = constant <f32: 1.000000e+00> : tile<1xf32>
    %ones = broadcast %one_1f : tile<1xf32> -> tile<1024xf32>
    %st = store_view_tko relaxed device %ones, %dp[%next] : tile<1024xf32>, partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
    %nslot = muli %next, %c32 : tile<i32>
    %one_1i = constant <i32: 1> : tile<1xi32>
    %ft = store_view_tko release device %one_1i, %fp[%nslot] token=%st : tile<1xi32>, partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> token

    // ---- 消费者角色：等自己的 flag，读自己独占的 chunk ----
    %myslot = muli %bx, %c32 : tile<i32>
    %lt = loop iter_values(%tok = %ft) : token -> token {
      %v, %t2 = load_view_tko relaxed device %fp[%myslot] token=%tok : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %vs = reshape %v : tile<1xi32> -> tile<i32>
      %rd = cmpi equal %vs, %c1, signed : tile<i32> -> tile<i1>
      if %rd { break %t2 : token }
      continue %t2 : token
    }
    %a, %at = load_view_tko acquire device %fp[%myslot] token=%lt : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token

    %d, %dt = load_view_tko relaxed device %dp[%bx] token=%at : partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
    %s = reduce %d dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
      (%e: tile<f32>, %ra: tile<f32>) { %z = addf %e, %ra : tile<f32>
        yield %z : tile<f32> }
    %r1 = reshape %s : tile<f32> -> tile<1xf32>
    %ov = make_tensor_view %out, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
    %op = make_partition_view %ov : partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>
    %ot = store_view_tko relaxed device %r1, %op[%bx] token=%dt : tile<1xf32>, partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
    return
  }
}
