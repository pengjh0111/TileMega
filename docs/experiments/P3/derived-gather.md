## Data-dependent gather

| # | edge | C | event shape | wait | fanout | volume | count | tier | attributes | guard | relaxation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | produce -> gather | `[S] -> { [m, n] -> [p0, p1] : m >= 0 and 128m < S and 0 <= n <= 31 and p0 >= 0 and 128p0 < S and 0 <= p1 <= 31 }` | `[]` | `[S] -> { [m, n] -> 32 * floor((127 + S)/128) : m >= 0 and 128m < S and 0 <= n <= 31 }` | `[S] -> { [p0, p1] -> 32 * floor((127 + S)/128) : p0 >= 0 and 128p0 < S and 0 <= p1 <= 31 }` | `{ 16384 }` | `[S] -> { 32 * floor((127 + S)/128) }` | 3 | data_dependent + symbolic_static + relaxed + tensor_values + uncountable | - | data-dependent index |

