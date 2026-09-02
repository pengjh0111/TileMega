## Llama decoder layer (one layer, symbolic S)

| # | edge | C | event shape | wait | fanout | volume | count | tier | guard | relaxation |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | rmsnorm1 -> wq | `(m,n) -> {rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 2 | rmsnorm1 -> wk | `(m,n) -> {rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_kv` | `(Tm * H)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 3 | rmsnorm1 -> wv | `(m,n) -> {rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_kv` | `(Tm * H)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 4 | wq -> rope_q | `(m,hh) -> {wq(m,hh)}` | `[ceildiv(S, Tm)xn_h]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 5 | wk -> rope_k | `(m,hh) -> {wk(m,hh)}` | `[ceildiv(S, Tm)xn_kv]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 6 | rope_k -> kvappend_k | `(row,hh) -> {rope_k(floordiv(row, Tm),hh)}` | `[Sxn_kv]` | `1` | `1` | `(Tm * d)` | `(S * n_kv)` | 1 | - | - |
| 7 | wv -> kvappend_v | `(row,hh) -> {wv(floordiv(row, Tm),hh)}` | `[Sxn_kv]` | `1` | `1` | `(Tm * d)` | `(S * n_kv)` | 1 | - | - |
| 8 | rope_q -> attn_chunk | `(s,h,j) -> {rope_q(floordiv(s, Tm),h)}` | `[Sxn_h]` | `1` | `ceildiv(L_s, Tkv)` | `(Tm * d)` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | - | - |
| 9 | kvappend_k -> attn_chunk | `(s,h,j) -> {kvappend_k(row,floordiv(h, G)) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 10 | kvappend_v -> attn_chunk | `(s,h,j) -> {kvappend_v(row,floordiv(h, G)) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 11 | attn_chunk -> attn_combine | `(s,h) -> {attn_chunk(s,h,j) : 0 <= j < 0 + ceildiv(L_s, Tkv)}` | `[Sxn_h]` | `ceildiv(L_s, Tkv)` | `1` | `d` | `(S * n_h)` | 2 | - | - |
| 12 | attn_combine -> wo | `(m,n) -> {attn_combine(s,h) : Tm*m <= s < Tm*m + Tm and 0 <= h < 0 + n_h}` | `[ceildiv(S, Tm)]` | `(Tm * n_h)` | `ceildiv(H, Tn)` | `d` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 13 | wo -> add1 | `(m,n) -> {wo(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 14 | add1 -> rmsnorm2 | `(i) -> {add1(i,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 15 | rmsnorm2 -> wgate | `(m,n) -> {rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 16 | rmsnorm2 -> wup | `(m,n) -> {rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 17 | wgate -> silu | `(m,n) -> {wgate(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 18 | wup -> silu | `(m,n) -> {wup(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 19 | silu -> wdown | `(m,n) -> {silu(m,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 20 | wdown -> add2 | `(m,n) -> {wdown(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 21 | add1 -> add2 | `(m,n) -> {add1(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |

