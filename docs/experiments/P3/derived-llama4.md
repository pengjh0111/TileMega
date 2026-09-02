## Llama stack, four layers

| # | edge | C | event shape | wait | fanout | volume | count | tier | guard | relaxation |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | l0.rmsnorm1 -> l0.wq | `(m,n) -> {l0.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 2 | l0.rmsnorm1 -> l0.wk | `(m,n) -> {l0.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_kv` | `(Tm * H)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 3 | l0.rmsnorm1 -> l0.wv | `(m,n) -> {l0.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_kv` | `(Tm * H)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 4 | l0.wq -> l0.rope_q | `(m,hh) -> {l0.wq(m,hh)}` | `[ceildiv(S, Tm)xn_h]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 5 | l0.wk -> l0.rope_k | `(m,hh) -> {l0.wk(m,hh)}` | `[ceildiv(S, Tm)xn_kv]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 6 | l0.rope_k -> l0.kvappend_k | `(row,hh) -> {l0.rope_k(floordiv(row, Tm),hh)}` | `[Sxn_kv]` | `1` | `1` | `(Tm * d)` | `(S * n_kv)` | 1 | - | - |
| 7 | l0.wv -> l0.kvappend_v | `(row,hh) -> {l0.wv(floordiv(row, Tm),hh)}` | `[Sxn_kv]` | `1` | `1` | `(Tm * d)` | `(S * n_kv)` | 1 | - | - |
| 8 | l0.rope_q -> l0.attn_chunk | `(s,h,j) -> {l0.rope_q(floordiv(s, Tm),h)}` | `[Sxn_h]` | `1` | `ceildiv(L_s, Tkv)` | `(Tm * d)` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | - | - |
| 9 | l0.kvappend_k -> l0.attn_chunk | `(s,h,j) -> {l0.kvappend_k(row,floordiv(h, G)) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 10 | l0.kvappend_v -> l0.attn_chunk | `(s,h,j) -> {l0.kvappend_v(row,floordiv(h, G)) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 11 | l0.attn_chunk -> l0.attn_combine | `(s,h) -> {l0.attn_chunk(s,h,j) : 0 <= j < 0 + ceildiv(L_s, Tkv)}` | `[Sxn_h]` | `ceildiv(L_s, Tkv)` | `1` | `d` | `(S * n_h)` | 2 | - | - |
| 12 | l0.attn_combine -> l0.wo | `(m,n) -> {l0.attn_combine(s,h) : Tm*m <= s < Tm*m + Tm and 0 <= h < 0 + n_h}` | `[ceildiv(S, Tm)]` | `(Tm * n_h)` | `ceildiv(H, Tn)` | `d` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 13 | l0.wo -> l0.add1 | `(m,n) -> {l0.wo(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 14 | l0.add1 -> l0.rmsnorm2 | `(i) -> {l0.add1(i,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 15 | l0.rmsnorm2 -> l0.wgate | `(m,n) -> {l0.rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 16 | l0.rmsnorm2 -> l0.wup | `(m,n) -> {l0.rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 17 | l0.wgate -> l0.silu | `(m,n) -> {l0.wgate(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 18 | l0.wup -> l0.silu | `(m,n) -> {l0.wup(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 19 | l0.silu -> l0.wdown | `(m,n) -> {l0.silu(m,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 20 | l0.wdown -> l0.add2 | `(m,n) -> {l0.wdown(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 21 | l0.add1 -> l0.add2 | `(m,n) -> {l0.add1(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 22 | l0.add2 -> l1.rmsnorm1 | `(m) -> {l0.add2(m,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 23 | l1.rmsnorm1 -> l1.wq | `(m,n) -> {l1.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 24 | l1.rmsnorm1 -> l1.wk | `(m,n) -> {l1.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_kv` | `(Tm * H)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 25 | l1.rmsnorm1 -> l1.wv | `(m,n) -> {l1.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_kv` | `(Tm * H)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 26 | l1.wq -> l1.rope_q | `(m,hh) -> {l1.wq(m,hh)}` | `[ceildiv(S, Tm)xn_h]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 27 | l1.wk -> l1.rope_k | `(m,hh) -> {l1.wk(m,hh)}` | `[ceildiv(S, Tm)xn_kv]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 28 | l1.rope_k -> l1.kvappend_k | `(row,hh) -> {l1.rope_k(floordiv(row, Tm),hh)}` | `[Sxn_kv]` | `1` | `1` | `(Tm * d)` | `(S * n_kv)` | 1 | - | - |
| 29 | l1.wv -> l1.kvappend_v | `(row,hh) -> {l1.wv(floordiv(row, Tm),hh)}` | `[Sxn_kv]` | `1` | `1` | `(Tm * d)` | `(S * n_kv)` | 1 | - | - |
| 30 | l1.rope_q -> l1.attn_chunk | `(s,h,j) -> {l1.rope_q(floordiv(s, Tm),h)}` | `[Sxn_h]` | `1` | `ceildiv(L_s, Tkv)` | `(Tm * d)` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | - | - |
| 31 | l1.kvappend_k -> l1.attn_chunk | `(s,h,j) -> {l1.kvappend_k(row,floordiv(h, G)) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 32 | l1.kvappend_v -> l1.attn_chunk | `(s,h,j) -> {l1.kvappend_v(row,floordiv(h, G)) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 33 | l1.attn_chunk -> l1.attn_combine | `(s,h) -> {l1.attn_chunk(s,h,j) : 0 <= j < 0 + ceildiv(L_s, Tkv)}` | `[Sxn_h]` | `ceildiv(L_s, Tkv)` | `1` | `d` | `(S * n_h)` | 2 | - | - |
| 34 | l1.attn_combine -> l1.wo | `(m,n) -> {l1.attn_combine(s,h) : Tm*m <= s < Tm*m + Tm and 0 <= h < 0 + n_h}` | `[ceildiv(S, Tm)]` | `(Tm * n_h)` | `ceildiv(H, Tn)` | `d` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 35 | l1.wo -> l1.add1 | `(m,n) -> {l1.wo(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 36 | l0.add2 -> l1.add1 | `(m,n) -> {l0.add2(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 37 | l1.add1 -> l1.rmsnorm2 | `(i) -> {l1.add1(i,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 38 | l1.rmsnorm2 -> l1.wgate | `(m,n) -> {l1.rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 39 | l1.rmsnorm2 -> l1.wup | `(m,n) -> {l1.rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 40 | l1.wgate -> l1.silu | `(m,n) -> {l1.wgate(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 41 | l1.wup -> l1.silu | `(m,n) -> {l1.wup(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 42 | l1.silu -> l1.wdown | `(m,n) -> {l1.silu(m,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 43 | l1.wdown -> l1.add2 | `(m,n) -> {l1.wdown(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 44 | l1.add1 -> l1.add2 | `(m,n) -> {l1.add1(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 45 | l1.add2 -> l2.rmsnorm1 | `(m) -> {l1.add2(m,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 46 | l2.rmsnorm1 -> l2.wq | `(m,n) -> {l2.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 47 | l2.rmsnorm1 -> l2.wk | `(m,n) -> {l2.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_kv` | `(Tm * H)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 48 | l2.rmsnorm1 -> l2.wv | `(m,n) -> {l2.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_kv` | `(Tm * H)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 49 | l2.wq -> l2.rope_q | `(m,hh) -> {l2.wq(m,hh)}` | `[ceildiv(S, Tm)xn_h]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 50 | l2.wk -> l2.rope_k | `(m,hh) -> {l2.wk(m,hh)}` | `[ceildiv(S, Tm)xn_kv]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 51 | l2.rope_k -> l2.kvappend_k | `(row,hh) -> {l2.rope_k(floordiv(row, Tm),hh)}` | `[Sxn_kv]` | `1` | `1` | `(Tm * d)` | `(S * n_kv)` | 1 | - | - |
| 52 | l2.wv -> l2.kvappend_v | `(row,hh) -> {l2.wv(floordiv(row, Tm),hh)}` | `[Sxn_kv]` | `1` | `1` | `(Tm * d)` | `(S * n_kv)` | 1 | - | - |
| 53 | l2.rope_q -> l2.attn_chunk | `(s,h,j) -> {l2.rope_q(floordiv(s, Tm),h)}` | `[Sxn_h]` | `1` | `ceildiv(L_s, Tkv)` | `(Tm * d)` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | - | - |
| 54 | l2.kvappend_k -> l2.attn_chunk | `(s,h,j) -> {l2.kvappend_k(row,floordiv(h, G)) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 55 | l2.kvappend_v -> l2.attn_chunk | `(s,h,j) -> {l2.kvappend_v(row,floordiv(h, G)) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 56 | l2.attn_chunk -> l2.attn_combine | `(s,h) -> {l2.attn_chunk(s,h,j) : 0 <= j < 0 + ceildiv(L_s, Tkv)}` | `[Sxn_h]` | `ceildiv(L_s, Tkv)` | `1` | `d` | `(S * n_h)` | 2 | - | - |
| 57 | l2.attn_combine -> l2.wo | `(m,n) -> {l2.attn_combine(s,h) : Tm*m <= s < Tm*m + Tm and 0 <= h < 0 + n_h}` | `[ceildiv(S, Tm)]` | `(Tm * n_h)` | `ceildiv(H, Tn)` | `d` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 58 | l2.wo -> l2.add1 | `(m,n) -> {l2.wo(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 59 | l1.add2 -> l2.add1 | `(m,n) -> {l1.add2(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 60 | l2.add1 -> l2.rmsnorm2 | `(i) -> {l2.add1(i,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 61 | l2.rmsnorm2 -> l2.wgate | `(m,n) -> {l2.rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 62 | l2.rmsnorm2 -> l2.wup | `(m,n) -> {l2.rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 63 | l2.wgate -> l2.silu | `(m,n) -> {l2.wgate(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 64 | l2.wup -> l2.silu | `(m,n) -> {l2.wup(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 65 | l2.silu -> l2.wdown | `(m,n) -> {l2.silu(m,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 66 | l2.wdown -> l2.add2 | `(m,n) -> {l2.wdown(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 67 | l2.add1 -> l2.add2 | `(m,n) -> {l2.add1(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 68 | l2.add2 -> l3.rmsnorm1 | `(m) -> {l2.add2(m,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 69 | l3.rmsnorm1 -> l3.wq | `(m,n) -> {l3.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_h` | `(Tm * H)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 70 | l3.rmsnorm1 -> l3.wk | `(m,n) -> {l3.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_kv` | `(Tm * H)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 71 | l3.rmsnorm1 -> l3.wv | `(m,n) -> {l3.rmsnorm1(m)}` | `[ceildiv(S, Tm)]` | `1` | `n_kv` | `(Tm * H)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 72 | l3.wq -> l3.rope_q | `(m,hh) -> {l3.wq(m,hh)}` | `[ceildiv(S, Tm)xn_h]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_h)` | 0 | - | - |
| 73 | l3.wk -> l3.rope_k | `(m,hh) -> {l3.wk(m,hh)}` | `[ceildiv(S, Tm)xn_kv]` | `1` | `1` | `(Tm * d)` | `(ceildiv(S, Tm) * n_kv)` | 0 | - | - |
| 74 | l3.rope_k -> l3.kvappend_k | `(row,hh) -> {l3.rope_k(floordiv(row, Tm),hh)}` | `[Sxn_kv]` | `1` | `1` | `(Tm * d)` | `(S * n_kv)` | 1 | - | - |
| 75 | l3.wv -> l3.kvappend_v | `(row,hh) -> {l3.wv(floordiv(row, Tm),hh)}` | `[Sxn_kv]` | `1` | `1` | `(Tm * d)` | `(S * n_kv)` | 1 | - | - |
| 76 | l3.rope_q -> l3.attn_chunk | `(s,h,j) -> {l3.rope_q(floordiv(s, Tm),h)}` | `[Sxn_h]` | `1` | `ceildiv(L_s, Tkv)` | `(Tm * d)` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | - | - |
| 77 | l3.kvappend_k -> l3.attn_chunk | `(s,h,j) -> {l3.kvappend_k(row,floordiv(h, G)) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 78 | l3.kvappend_v -> l3.attn_chunk | `(s,h,j) -> {l3.kvappend_v(row,floordiv(h, G)) : Tkv*j + (-1 * past) <= row < Tkv*j + (-1 * past) + Tkv}` | `[n_hxceildiv(L_s, Tkv)]` | `Tkv` | `S` | `d` | `((S * n_h) * ceildiv(L_s, Tkv))` | 2 | `j == floordiv(past, Tkv)` | - |
| 79 | l3.attn_chunk -> l3.attn_combine | `(s,h) -> {l3.attn_chunk(s,h,j) : 0 <= j < 0 + ceildiv(L_s, Tkv)}` | `[Sxn_h]` | `ceildiv(L_s, Tkv)` | `1` | `d` | `(S * n_h)` | 2 | - | - |
| 80 | l3.attn_combine -> l3.wo | `(m,n) -> {l3.attn_combine(s,h) : Tm*m <= s < Tm*m + Tm and 0 <= h < 0 + n_h}` | `[ceildiv(S, Tm)]` | `(Tm * n_h)` | `ceildiv(H, Tn)` | `d` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 81 | l3.wo -> l3.add1 | `(m,n) -> {l3.wo(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 82 | l2.add2 -> l3.add1 | `(m,n) -> {l2.add2(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 83 | l3.add1 -> l3.rmsnorm2 | `(i) -> {l3.add1(i,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 84 | l3.rmsnorm2 -> l3.wgate | `(m,n) -> {l3.rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 85 | l3.rmsnorm2 -> l3.wup | `(m,n) -> {l3.rmsnorm2(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 86 | l3.wgate -> l3.silu | `(m,n) -> {l3.wgate(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 87 | l3.wup -> l3.silu | `(m,n) -> {l3.wup(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 88 | l3.silu -> l3.wdown | `(m,n) -> {l3.silu(m,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 89 | l3.wdown -> l3.add2 | `(m,n) -> {l3.wdown(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 90 | l3.add1 -> l3.add2 | `(m,n) -> {l3.add1(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |

