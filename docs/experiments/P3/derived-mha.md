## MHA model (no GQA), two layers

| # | edge | C | event shape | wait | fanout | volume | count | tier | guard | relaxation |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | mha0.rmsnorm1 -> mha0.wq | `(m,n) -> {mha0.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 2 | mha0.rmsnorm1 -> mha0.wk | `(m,n) -> {mha0.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 3 | mha0.rmsnorm1 -> mha0.wv | `(m,n) -> {mha0.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 4 | mha0.wq -> mha0.rope_q | `(m,hh) -> {mha0.wq(m,hh)}` | `[ceildiv(S, Tm)xn_h]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 5 | mha0.wk -> mha0.rope_k | `(m,hh) -> {mha0.wk(m,hh)}` | `[ceildiv(S, Tm)xn_h]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 6 | mha0.rope_k -> mha0.kvappend_k | `(row,hh) -> {mha0.rope_k(floordiv(row, Tm),hh)}` | `[Sxn_h]` | `1` | `1` | `(Tm * d)` | `(S * n_h)` | 1 | - | - |
| 7 | mha0.wv -> mha0.kvappend_v | `(row,hh) -> {mha0.wv(floordiv(row, Tm),hh)}` | `[Sxn_h]` | `1` | `1` | `(Tm * d)` | `(S * n_h)` | 1 | - | - |
| 8 | mha0.rope_q -> mha0.attn_chunk | `(s,h,j) -> {mha0.rope_q(floordiv(s, Tm),h)}` | `[Sxn_h]` | `1` | `ceildiv(L_s, Tkv)` | `(Tm * d)` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | - | - |
| 9 | mha0.kvappend_k -> mha0.attn_chunk | `(s,h,j) -> {mha0.kvappend_k(row,h) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 10 | mha0.kvappend_v -> mha0.attn_chunk | `(s,h,j) -> {mha0.kvappend_v(row,h) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 11 | mha0.attn_chunk -> mha0.attn_combine | `(s,h) -> {mha0.attn_chunk(s,h,j) : 0 <= j < 0 + ceildiv(L_s, Tkv)}` | `[Sxn_h]` | `ceildiv(L_s, Tkv)` | `1` | `d` | `(S * n_h)` | 2 | - | - |
| 12 | mha0.attn_combine -> mha0.wo | `(m,n) -> {mha0.attn_combine(s,h) : Tm*m <= s < Tm*m + Tm and 0 <= h < 0 + n_h}` | `[ceildiv(S, Tm)]` | `(Tm * n_h)` | `ceildiv(H, Tn)` | `d` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 13 | mha0.wo -> mha0.add1 | `(m,n) -> {mha0.wo(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 14 | mha0.add1 -> mha0.rmsnorm2 | `(i) -> {mha0.add1(i,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 15 | mha0.rmsnorm2 -> mha0.wup | `(m,n) -> {mha0.rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 16 | mha0.wup -> mha0.wdown | `(m,n) -> {mha0.wup(m,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 17 | mha0.wdown -> mha0.add2 | `(m,n) -> {mha0.wdown(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 18 | mha0.add1 -> mha0.add2 | `(m,n) -> {mha0.add1(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 19 | mha1.rmsnorm1 -> mha1.wq | `(m,n) -> {mha1.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 20 | mha1.rmsnorm1 -> mha1.wk | `(m,n) -> {mha1.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 21 | mha1.rmsnorm1 -> mha1.wv | `(m,n) -> {mha1.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 22 | mha1.wq -> mha1.rope_q | `(m,hh) -> {mha1.wq(m,hh)}` | `[ceildiv(S, Tm)xn_h]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 23 | mha1.wk -> mha1.rope_k | `(m,hh) -> {mha1.wk(m,hh)}` | `[ceildiv(S, Tm)xn_h]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 24 | mha1.rope_k -> mha1.kvappend_k | `(row,hh) -> {mha1.rope_k(floordiv(row, Tm),hh)}` | `[Sxn_h]` | `1` | `1` | `(Tm * d)` | `(S * n_h)` | 1 | - | - |
| 25 | mha1.wv -> mha1.kvappend_v | `(row,hh) -> {mha1.wv(floordiv(row, Tm),hh)}` | `[Sxn_h]` | `1` | `1` | `(Tm * d)` | `(S * n_h)` | 1 | - | - |
| 26 | mha1.rope_q -> mha1.attn_chunk | `(s,h,j) -> {mha1.rope_q(floordiv(s, Tm),h)}` | `[Sxn_h]` | `1` | `ceildiv(L_s, Tkv)` | `(Tm * d)` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | - | - |
| 27 | mha1.kvappend_k -> mha1.attn_chunk | `(s,h,j) -> {mha1.kvappend_k(row,h) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 28 | mha1.kvappend_v -> mha1.attn_chunk | `(s,h,j) -> {mha1.kvappend_v(row,h) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 29 | mha1.attn_chunk -> mha1.attn_combine | `(s,h) -> {mha1.attn_chunk(s,h,j) : 0 <= j < 0 + ceildiv(L_s, Tkv)}` | `[Sxn_h]` | `ceildiv(L_s, Tkv)` | `1` | `d` | `(S * n_h)` | 2 | - | - |
| 30 | mha1.attn_combine -> mha1.wo | `(m,n) -> {mha1.attn_combine(s,h) : Tm*m <= s < Tm*m + Tm and 0 <= h < 0 + n_h}` | `[ceildiv(S, Tm)]` | `(Tm * n_h)` | `ceildiv(H, Tn)` | `d` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 31 | mha1.wo -> mha1.add1 | `(m,n) -> {mha1.wo(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 32 | mha1.add1 -> mha1.rmsnorm2 | `(i) -> {mha1.add1(i,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 33 | mha1.rmsnorm2 -> mha1.wup | `(m,n) -> {mha1.rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 34 | mha1.wup -> mha1.wdown | `(m,n) -> {mha1.wup(m,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 35 | mha1.wdown -> mha1.add2 | `(m,n) -> {mha1.wdown(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 36 | mha1.add1 -> mha1.add2 | `(m,n) -> {mha1.add1(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |

