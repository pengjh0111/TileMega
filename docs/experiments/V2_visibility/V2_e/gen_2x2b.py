#!/usr/bin/env python3
"""2x2 factorial on the V1_min SHAPE (consumers spin immediately).

The first 2x2 (gen_2x2.py) was built on the V2-e chain, where every block does a
1024-float store BEFORE it spins. That store delays each consumer past the moment
the flag flips, so no consumer ever really spins -- all four cells passed for a
reason that had nothing to do with the factors under test. This version removes
that confound:

  block 0        pure producer: writes chunks [0, grid) in a loop, then releases
  block bx >= 1  pure consumer: spins IMMEDIATELY, then reduces one chunk

Factors:
  flag = shared  -> every consumer spins on flag[0]         (V1-a / V1-b shape)
  flag = dist    -> consumer bx spins on flag[bx*32]        (V1-d shape)
  data = shared  -> every consumer reduces chunk 0          (V1-b shape, FAILS)
  data = excl    -> consumer bx reduces chunk bx            (V1-a shape, PASSES)
"""
TMPL = r'''
cuda_tile.module @cuda_tile_module {{
  entry @{name}(%data: tile<ptr<f32>>, %flag: tile<ptr<i32>>, %out: tile<ptr<f32>>) {{
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
    if %is_prod {{
      %one_1f = constant <f32: 1.000000e+00> : tile<1xf32>
      %ones = broadcast %one_1f : tile<1xf32> -> tile<1024xf32>
      %init = make_token : token
      %chain = for %c in (%c0 to %ngrid, step %c1) : tile<i32>
                   iter_values(%tok = %init) -> (token) {{
        %t = store_view_tko relaxed device %ones, %dp[%c] token=%tok : tile<1024xf32>, partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
        continue %t : token
      }}
{release}
      yield
    }} else {{
      %myslot = {waitslot}
      %it = make_token : token
      %lt = loop iter_values(%tok = %it) : token -> token {{
        %v, %t2 = load_view_tko relaxed device %fp[%myslot] token=%tok : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
        %vs = reshape %v : tile<1xi32> -> tile<i32>
        %rd = cmpi equal %vs, %c1, signed : tile<i32> -> tile<i1>
        if %rd {{ break %t2 : token }}
        continue %t2 : token
      }}
      %a, %at = load_view_tko acquire device %fp[%myslot] token=%lt : partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> tile<1xi32>, token
      %d, %dt = load_view_tko relaxed device %dp[{dataidx}] token=%at : partition_view<tile=(1024), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      %s = reduce %d dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
        (%e: tile<f32>, %ra: tile<f32>) {{ %z = addf %e, %ra : tile<f32>
          yield %z : tile<f32> }}
      %r1 = reshape %s : tile<f32> -> tile<1xf32>
      %ov = make_tensor_view %out, shape = [524288], strides = [1] : tensor_view<524288xf32, strides=[1]>
      %op = make_partition_view %ov : partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>
      %ot = store_view_tko relaxed device %r1, %op[%bx] token=%dt : tile<1xf32>, partition_view<tile=(1), tensor_view<524288xf32, strides=[1]>>, tile<i32> -> token
      yield
    }}
    return
  }}
}}
'''
REL_SHARED = '''      %ft = store_view_tko release device %one_1i, %fp[%c0] token=%chain : tile<1xi32>, partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> token'''
REL_DIST = '''      %ftok = for %j in (%c1 to %ngrid, step %c1) : tile<i32>
                  iter_values(%tk = %chain) -> (token) {
        %fidx = muli %j, %c32 : tile<i32>
        %ft = store_view_tko release device %one_1i, %fp[%fidx] token=%tk : tile<1xi32>, partition_view<tile=(1), tensor_view<16384xi32, strides=[1]>>, tile<i32> -> token
        continue %ft : token
      }'''
import subprocess
for fl in ("shared", "dist"):
    for dt in ("excl", "shared"):
        name = f"v2_y_{fl}_{dt}"
        src = TMPL.format(name=name,
                          release=REL_SHARED if fl == "shared" else REL_DIST,
                          waitslot="addi %c0, %c0 : tile<i32>" if fl == "shared" else "muli %bx, %c32 : tile<i32>",
                          dataidx="%c0" if dt == "shared" else "%bx")
        open(f"{name}.mlir","w").write(src)
        r = subprocess.run(["/data/tilemega/scripts/tilemega-compile","-o",f"{name}.cubin",f"{name}.mlir"],
                           capture_output=True, text=True)
        print(f"{name:22s} {'OK' if r.returncode==0 else 'FAIL'}")
        if r.returncode: print(r.stdout[-500:], r.stderr[-500:])
