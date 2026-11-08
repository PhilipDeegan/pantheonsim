| program | tier A s (spread) | tier again A' s | JIT s (spread) | **A/JIT** | A/A' | warp instructions M | ran in regions % | computed by generated code % | region calls M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/SYRK | 5.40 (1%) | 5.34 | 4.80 (3%) | **1.125** | 1.010 | 194 | 100 | 47 | 0.10 |
| polybench/SYR2K | 10.20 (5%) | 10.32 | 8.97 (4%) | **1.136** | 0.988 | 399 | 100 | 57 | 0.10 |
| samples/convolutionSeparable | 11.39 (2%) | 11.55 | 11.41 (21%) | **0.998** | 0.987 | 385 | 0 | 0 | 0 |
| samples/mergeSort | 8.50 (2%) | 8.50 | 7.80 (2%) | **1.090** | 1.000 | 354 | 73 | 53 | 20.39 |
| samples/sortingNetworks | 13.67 (0%) | 13.66 | 12.90 (1%) | **1.060** | 1.001 | 492 | 44 | 33 | 8.26 |
| hecbench/fpdc-cuda | 236.96 (34%) | - | 241.25 (21%) | **0.982** | - | 258 | 67 | 52 | 18.75 |
| samples/interval | 15.87 (3%) | - | 14.54 (1%) | **1.091** | - | 795 | 81 | 49 | 36.13 |
| samples/histogram | 13.70 (1%) | - | 12.37 (1%) | **1.107** | - | 448 | 80 | 61 | 9.29 |
| hecbench/black-scholes-cuda | 20.16 (7%) | - | 20.50 (12%) | **0.984** | - | 1222 | 75 | 65 | 115.61 |
| samples/jacobiCudaGraphs | 19.03 (1%) | - | 18.94 (1%) | **1.005** | - | 916 | 0 | 0 | 0 |
| polybench/LU | 29.91 (1%) | - | 25.34 (3%) | **1.180** | - | 1777 | 90 | 82 | 136.62 |
| samples/simpleHyperQ | 11.94 (1%) | - | 7.02 (2%) | **1.701** | - | 1176 | 100 | 40 | 0.11 |
| samples/alignedTypes | 50.19 (2%) | - | 45.86 (2%) | **1.094** | - | 972 | 72 | 45 | 0.50 |
| samples/radixSortThrust | 39.52 (1%) | - | 39.79 (0%) | **0.993** | - | 1387 | 18 | 15 | 32.33 |
| polybench/FDTD-2D | 99.69 (1%) | - | 85.45 (2%) | **1.167** | - | 5373 | 89 | 74 | 393.19 |
| hecbench/shmembench-cuda | 4.73 (1%) | - | 4.75 (2%) | **0.996** | - | 186 | 0 | 0 | 0 |
| samples/reduction | 9.54 (1%) | - | 7.61 (1%) | **1.253** | - | 348 | 100 | 77 | 0.15 |
| samples/transpose | 8.18 (4%) | - | 7.01 (5%) | **1.166** | - | 427 | 80 | 68 | 13.21 |
| samples/segmentationTreeThrust | 33.05 (20%) | - | 30.23 (12%) | **1.093** | - | 610 | 64 | 56 | 61.41 |
| samples/cudaTensorCoreGemm | 13.55 (5%) | - | 13.47 (0%) | **1.006** | - | 59 | 0 | 0 | 0 |
| hecbench/atomicAggregate-cuda | 4.73 (2%) | - | 4.35 (3%) | **1.088** | - | 164 | 78 | 60 | 14.67 |
| rodinia/gaussian | 3.67 (1%) | - | 3.53 (2%) | **1.039** | - | 165 | 89 | 84 | 15.45 |

programs 22; geomean A/JIT **1.0989**; worst 0.982 (hecbench/fpdc-cuda); best 1.701 (samples/simpleHyperQ); slower than -2%: none; sum A 664 s, sum JIT 628 s (1.057x)
A/A' over 5 programs: 0.987 to 1.010 (geomean 0.9971)
output hashes: DIFFER in samples/segmentationTreeThrust

