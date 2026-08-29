import sys

# R2-G: generate an N-way heterogeneous dispatch kernel (nested if/else chain since
# cuda_tile only has binary `if`), each branch doing a genuinely different tile
# shape/op, to test whether resource usage stays MAX-like or degrades toward SUM
# as branch count grows. Branch "workloads" cycle through a fixed palette so that
# N=8 reuses the same 8 distinct ops regardless of N (for comparability).

WORKLOADS = [
    # (label, mlir_snippet_generator)
    ("mm128", lambda i: f"""
      %tview_a{i} = make_tensor_view %in_a, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
      %pview_a{i} = make_partition_view %tview_a{i} : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
      %A{i}, %tok_a{i} = load_view_tko weak %pview_a{i}[%c0_i32, %c0_i32] : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> tile<128x128xf32>, token
      %tview_b{i} = make_tensor_view %in_b, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
      %pview_b{i} = make_partition_view %tview_b{i} : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
      %B{i}, %tok_b{i} = load_view_tko weak %pview_b{i}[%c0_i32, %c0_i32] : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> tile<128x128xf32>, token
      %zero_mat{i} = constant <f32: 0.000000e+00> : tile<128x128xf32>
      %C{i} = mmaf %A{i}, %B{i}, %zero_mat{i} : tile<128x128xf32>, tile<128x128xf32>, tile<128x128xf32>
      %tview_out{i} = make_tensor_view %out, shape = [128, 128], strides = [128, 1] : tensor_view<128x128xf32, strides=[128,1]>
      %pview_out{i} = make_partition_view %tview_out{i} : partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>
      %tok_out{i} = store_view_tko weak %C{i}, %pview_out{i}[%c0_i32, %c0_i32] : tile<128x128xf32>, partition_view<tile=(128x128), tensor_view<128x128xf32, strides=[128,1]>>, tile<i32> -> token
"""),
    ("red256", lambda i: f"""
      %tview_v{i} = make_tensor_view %in_a, shape = [256], strides = [1] : tensor_view<256xf32, strides=[1]>
      %pview_v{i} = make_partition_view %tview_v{i} : partition_view<tile=(256), tensor_view<256xf32, strides=[1]>>
      %vec{i}, %tok_v{i} = load_view_tko weak %pview_v{i}[%c0_i32] : partition_view<tile=(256), tensor_view<256xf32, strides=[1]>>, tile<i32> -> tile<256xf32>, token
      %sum{i} = reduce %vec{i} dim=0 identities=[0.000000e+00 : f32] : tile<256xf32> -> tile<f32>
        (%elem{i}: tile<f32>, %acc{i}: tile<f32>) {{
          %add{i} = addf %elem{i}, %acc{i} : tile<f32>
          yield %add{i} : tile<f32>
        }}
      %sum_1d{i} = reshape %sum{i} : tile<f32> -> tile<1xf32>
      %tview_out{i} = make_tensor_view %out, shape = [1], strides = [1] : tensor_view<1xf32, strides=[1]>
      %pview_out{i} = make_partition_view %tview_out{i} : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>
      %tok_out{i} = store_view_tko weak %sum_1d{i}, %pview_out{i}[%c0_i32] : tile<1xf32>, partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>, tile<i32> -> token
"""),
    ("mm64", lambda i: f"""
      %tview_a{i} = make_tensor_view %in_a, shape = [64, 64], strides = [64, 1] : tensor_view<64x64xf32, strides=[64,1]>
      %pview_a{i} = make_partition_view %tview_a{i} : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>
      %A{i}, %tok_a{i} = load_view_tko weak %pview_a{i}[%c0_i32, %c0_i32] : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>, tile<i32> -> tile<64x64xf32>, token
      %tview_b{i} = make_tensor_view %in_b, shape = [64, 64], strides = [64, 1] : tensor_view<64x64xf32, strides=[64,1]>
      %pview_b{i} = make_partition_view %tview_b{i} : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>
      %B{i}, %tok_b{i} = load_view_tko weak %pview_b{i}[%c0_i32, %c0_i32] : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>, tile<i32> -> tile<64x64xf32>, token
      %zero_mat{i} = constant <f32: 0.000000e+00> : tile<64x64xf32>
      %C{i} = mmaf %A{i}, %B{i}, %zero_mat{i} : tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>
      %tview_out{i} = make_tensor_view %out, shape = [64, 64], strides = [64, 1] : tensor_view<64x64xf32, strides=[64,1]>
      %pview_out{i} = make_partition_view %tview_out{i} : partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>
      %tok_out{i} = store_view_tko weak %C{i}, %pview_out{i}[%c0_i32, %c0_i32] : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<64x64xf32, strides=[64,1]>>, tile<i32> -> token
"""),
    ("red1024", lambda i: f"""
      %tview_v{i} = make_tensor_view %in_a, shape = [1024], strides = [1] : tensor_view<1024xf32, strides=[1]>
      %pview_v{i} = make_partition_view %tview_v{i} : partition_view<tile=(1024), tensor_view<1024xf32, strides=[1]>>
      %vec{i}, %tok_v{i} = load_view_tko weak %pview_v{i}[%c0_i32] : partition_view<tile=(1024), tensor_view<1024xf32, strides=[1]>>, tile<i32> -> tile<1024xf32>, token
      %sum{i} = reduce %vec{i} dim=0 identities=[0.000000e+00 : f32] : tile<1024xf32> -> tile<f32>
        (%elem{i}: tile<f32>, %acc{i}: tile<f32>) {{
          %add{i} = addf %elem{i}, %acc{i} : tile<f32>
          yield %add{i} : tile<f32>
        }}
      %sum_1d{i} = reshape %sum{i} : tile<f32> -> tile<1xf32>
      %tview_out{i} = make_tensor_view %out, shape = [1], strides = [1] : tensor_view<1xf32, strides=[1]>
      %pview_out{i} = make_partition_view %tview_out{i} : partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>
      %tok_out{i} = store_view_tko weak %sum_1d{i}, %pview_out{i}[%c0_i32] : tile<1xf32>, partition_view<tile=(1), tensor_view<1xf32, strides=[1]>>, tile<i32> -> token
"""),
]

def gen(n, entry_name):
    lines = []
    lines.append(f"cuda_tile.module @cuda_tile_module {{")
    lines.append(f"  entry @{entry_name}(%task_type: tile<i32>, %in_a: tile<ptr<f32>>, %in_b: tile<ptr<f32>>, %out: tile<ptr<f32>>) {{")
    lines.append("    %c0_i32 = constant <i32: 0> : tile<i32>")
    for i in range(n):
        indent = "    " + "  " * i
        if i < n - 1:
            if i > 0:
                lines.append(f"{indent}%c{i}_i32 = constant <i32: {i}> : tile<i32>")
            lines.append(f"{indent}%cond{i} = cmpi equal %task_type, %c{i}_i32, signed : tile<i32> -> tile<i1>")
            lines.append(f"{indent}if %cond{i} {{")
            label, gen_fn = WORKLOADS[i % len(WORKLOADS)]
            lines.append(gen_fn(i))
            lines.append(f"{indent}  yield")
            lines.append(f"{indent}}} else {{")
        else:
            label, gen_fn = WORKLOADS[i % len(WORKLOADS)]
            lines.append(gen_fn(i))
            lines.append(f"{indent}yield")
    for i in range(n - 1):
        indent = "    " + "  " * (n - 2 - i)
        lines.append(f"{indent}}}")
    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)

if __name__ == "__main__":
    n = int(sys.argv[1])
    entry_name = sys.argv[2]
    print(gen(n, entry_name))
