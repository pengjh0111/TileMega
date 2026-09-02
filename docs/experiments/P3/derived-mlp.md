## MLP stack, three blocks

| # | edge | C | event shape | wait | fanout | volume | count | tier | guard | relaxation |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | b0.norm -> b0.fc1 | `(m,n) -> {b0.norm(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 2 | b0.fc1 -> b0.gelu | `(m,n) -> {b0.fc1(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 3 | b0.gelu -> b0.fc2 | `(m,n) -> {b0.gelu(m,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 4 | b0.fc2 -> b0.add | `(m,n) -> {b0.fc2(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 5 | b0.add -> b1.norm | `(m) -> {b0.add(m,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 6 | b1.norm -> b1.fc1 | `(m,n) -> {b1.norm(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 7 | b1.fc1 -> b1.gelu | `(m,n) -> {b1.fc1(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 8 | b1.gelu -> b1.fc2 | `(m,n) -> {b1.gelu(m,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 9 | b1.fc2 -> b1.add | `(m,n) -> {b1.fc2(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 10 | b0.add -> b1.add | `(m,n) -> {b0.add(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 11 | b1.add -> b2.norm | `(m) -> {b1.add(m,n) : 0 <= n < 0 + ceildiv(H, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(H, Tn)` | `1` | `(Tm * Tn)` | `ceildiv(S, Tm)` | 0 | - | - |
| 12 | b2.norm -> b2.fc1 | `(m,n) -> {b2.norm(m)}` | `[ceildiv(S, Tm)]` | `1` | `ceildiv(I, Tn)` | `(Tm * H)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 13 | b2.fc1 -> b2.gelu | `(m,n) -> {b2.fc1(m,n)}` | `[ceildiv(S, Tm)xceildiv(I, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(I, Tn))` | 0 | - | - |
| 14 | b2.gelu -> b2.fc2 | `(m,n) -> {b2.gelu(m,n_p1) : 0 <= n_p1 < 0 + ceildiv(I, Tn)}` | `[ceildiv(S, Tm)]` | `ceildiv(I, Tn)` | `ceildiv(H, Tn)` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 15 | b2.fc2 -> b2.add | `(m,n) -> {b2.fc2(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |
| 16 | b1.add -> b2.add | `(m,n) -> {b1.add(m,n)}` | `[ceildiv(S, Tm)xceildiv(H, Tn)]` | `1` | `1` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 0 | - | - |

