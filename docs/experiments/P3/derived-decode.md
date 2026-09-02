## Llama decoder layer (decode instantiation, S = 1)

| # | edge | C | event shape | wait | fanout | volume | count | tier | guard | relaxation |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rmsnorm1 -> wq | `(m,n) -> {rmsnorm1(0)}` | `[]` | `1` | `n_h` | `(Tm * H)` | `n_h` | 0 | - | - |
| 2 | rmsnorm1 -> wk | `(m,n) -> {rmsnorm1(0)}` | `[]` | `1` | `n_kv` | `(Tm * H)` | `n_kv` | 0 | - | - |
| 3 | rmsnorm1 -> wv | `(m,n) -> {rmsnorm1(0)}` | `[]` | `1` | `n_kv` | `(Tm * H)` | `n_kv` | 0 | - | - |
| 4 | wq -> rope_q | `(m,hh) -> {wq(0,hh)}` | `[n_h]` | `1` | `1` | `(Tm * d)` | `n_h` | 0 | - | - |
| 5 | wk -> rope_k | `(m,hh) -> {wk(0,hh)}` | `[n_kv]` | `1` | `1` | `(Tm * d)` | `n_kv` | 0 | - | - |
| 6 | rope_k -> kvappend_k | `(hh) -> {rope_k(0,hh)}` | `[n_kv]` | `1` | `1` | `(Tm * d)` | `n_kv` | 1 | - | - |
| 7 | wv -> kvappend_v | `(hh) -> {wv(0,hh)}` | `[n_kv]` | `1` | `1` | `(Tm * d)` | `n_kv` | 1 | - | - |
| 8 | rope_q -> attn_chunk | `(h,j) -> {rope_q(0,h)}` | `[n_h]` | `1` | `ceildiv(L_s, Tkv)` | `(Tm * d)` | `(n_h * ceildiv(L_s, Tkv))` | 2 | - | - |
| 9 | kvappend_k -> attn_chunk | `(h,j) -> {kvappend_k(floordiv(h, G))}` | `[n_h]` | `1` | `ceildiv(L_s, Tkv)` | `d` | `(n_h * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 10 | kvappend_v -> attn_chunk | `(h,j) -> {kvappend_v(floordiv(h, G))}` | `[n_h]` | `1` | `ceildiv(L_s, Tkv)` | `d` | `(n_h * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 11 | attn_chunk -> attn_combine | `(h) -> {attn_chunk(h,j) : 0 <= j < 0 + ceildiv(L_s, Tkv)}` | `[n_h]` | `ceildiv(L_s, Tkv)` | `1` | `d` | `n_h` | 2 | - | - |
| 12 | attn_combine -> wo | `(m,n) -> {attn_combine(h) : 0 <= h < 0 + n_h}` | `[]` | `n_h` | `ceildiv(H, Tn)` | `d` | `ceildiv(H, Tn)` | 0 | - | - |
| 13 | wo -> add1 | `(m,n) -> {wo(0,n)}` | `[ceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `ceildiv(H, Tn)` | 0 | - | - |
| 14 | add1 -> rmsnorm2 | `(i) -> {add1(0,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `1` | 0 | - | - |
| 15 | rmsnorm2 -> wgate | `(m,n) -> {rmsnorm2(0)}` | `[]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `ceildiv(I, Tn)` | 0 | - | - |
| 16 | rmsnorm2 -> wup | `(m,n) -> {rmsnorm2(0)}` | `[]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `ceildiv(I, Tn)` | 0 | - | - |
| 17 | wgate -> silu | `(m,n) -> {wgate(0,n)}` | `[ceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `ceildiv(I, Tn)` | 0 | - | - |
| 18 | wup -> silu | `(m,n) -> {wup(0,n)}` | `[ceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `ceildiv(I, Tn)` | 0 | - | - |
| 19 | silu -> wdown | `(m,n) -> {silu(0,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `ceildiv(H, Tn)` | 0 | - | - |
| 20 | wdown -> add2 | `(m,n) -> {wdown(0,n)}` | `[ceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `ceildiv(H, Tn)` | 0 | - | - |
| 21 | add1 -> add2 | `(m,n) -> {add1(0,n)}` | `[ceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `ceildiv(H, Tn)` | 0 | - | - |

