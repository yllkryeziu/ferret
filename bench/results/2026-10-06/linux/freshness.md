# Freshness: edit to searchable

The first 1 trial(s) of each path are warm-up and left out of every row below: Ferret Notify 10.88 ms; Ferret watcher 12.03 ms; Zoekt commit + delta + reload 15629.94 ms.

| Path | n | p50 (ms) | p90 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|---:|
| Ferret Notify | 100 | 0.33 | 0.70 | 2.18 | 2.18 |
| Ferret watcher | 100 | 11.35 | 11.58 | 12.19 | 12.19 |
| Zoekt commit + delta + reload | 100 | 260.77 | 330.07 | 1061.98 | 1061.98 |

## Stages (p50 / p99, ms)

| Path | Stage | p50 | p99 | max |
|---|---|---:|---:|---:|
| Ferret Notify | Notify call | 0.01 | 0.01 | 0.01 |
| Ferret Notify | wait for the generation | 0.07 | 0.31 | 0.31 |
| Ferret Notify | search that sees it | 0.16 | 2.13 | 2.13 |
| Ferret watcher | write to Ferret's callback (FSEvents) | 11.02 | 11.81 | 11.81 |
| Ferret watcher | callback to applied | 0.11 | 0.30 | 0.30 |
| Ferret watcher | applied to seen by the poll | 0.24 | 0.40 | 0.40 |
| Zoekt commit + delta + reload | git commit | 183.02 | 547.15 | 547.15 |
| Zoekt commit + delta + reload | delta build | 28.93 | 359.72 | 359.72 |
| Zoekt commit + delta + reload | reload and search | 44.07 | 286.46 | 286.46 |
