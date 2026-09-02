## Data-dependent gather

| # | edge | C | event shape | wait | fanout | volume | count | tier | guard | relaxation |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | produce -> gather | `(m,n) -> {produce(m_p1,n_p1) : 0 <= m_p1 < 0 + ceildiv(S, Tm) and 0 <= n_p1 < 0 + ceildiv(H, Tn)}` | `[]` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | `(Tm * Tn)` | `(ceildiv(S, Tm) * ceildiv(H, Tn))` | 3 | - | data-dependent index |

