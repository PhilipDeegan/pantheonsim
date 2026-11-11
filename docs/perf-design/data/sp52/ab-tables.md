A/B, warm cache, 5 interleaved runs a side, medians of whole-process wall seconds (spread = (max - min) / median); sides of one build: JIT, JIT again (the control), JIT with VGPU_JIT_LINK=1.

| program | JIT s (spread) | JIT again s (spread) | linked s (spread) | **JIT / linked** | control JIT / JIT again | outputs |
|---|---:|---:|---:|---:|---:|---|
| hecbench/black-scholes-cuda | 19.666 (1%) | 19.661 (1%) | 16.261 (1%) | **1.209** | 1.000 | same |
| polybench/LU | 24.528 (4%) | 24.143 (6%) | 22.825 (2%) | **1.075** | 1.016 | same |
| rodinia/gaussian | 3.373 (1%) | 3.362 (1%) | 3.178 (1%) | **1.061** | 1.003 | same |
| rodinia/hotspot | 0.872 (16%) | 0.869 (1%) | 0.782 (3%) | **1.116** | 1.004 | same |
| polybench/SYRK | 4.917 (2%) | 4.900 (4%) | 4.909 (2%) | **1.002** | 1.003 | same |

geometric mean JIT / linked: **1.0905** over 5 programs (1.002 to 1.209); control: 1.0054 (1.000 to 1.016)

| program | JIT s (spread) | JIT again s (spread) | linked s (spread) | **JIT / linked** | control JIT / JIT again | outputs |
|---|---:|---:|---:|---:|---:|---|
| polybench/SYR2K | 9.131 (1%) | 9.141 (12%) | 9.442 (8%) | **0.967** | 0.999 | same |
| polybench/JACOBI1D | 0.938 (1%) | 0.938 (1%) | 0.830 (2%) | **1.131** | 1.000 | same |
| polybench/JACOBI2D | 0.496 (34%) | 0.494 (6%) | 0.463 (0%) | **1.073** | 1.005 | same |
| polybench/3DCONV | 0.614 (2%) | 0.615 (5%) | 0.608 (3%) | **1.010** | 0.997 | same |
| polybench/2DCONV | 0.654 (2%) | 0.657 (4%) | 0.651 (1%) | **1.005** | 0.995 | same |
| rodinia/lud | 2.925 (1%) | 2.912 (0%) | 2.910 (0%) | **1.005** | 1.004 | same |
| rodinia/backprop | 0.517 (2%) | 0.511 (19%) | 0.510 (1%) | **1.014** | 1.011 | same |
| rodinia/pathfinder | 0.509 (3%) | 0.512 (2%) | 0.512 (3%) | **0.995** | 0.995 | same |
| hecbench/bilateral-cuda | 1.092 (1%) | 1.093 (2%) | 0.984 (6%) | **1.109** | 0.999 | same |
| hecbench/bh-cuda | 0.618 (1%) | 0.616 (1%) | 0.587 (1%) | **1.053** | 1.003 | same |
| hecbench/silu-cuda | 0.388 (1%) | 0.389 (1%) | 0.366 (3%) | **1.060** | 0.998 | same |
| hecbench/aop-cuda | 1.802 (1%) | 1.790 (1%) | 1.791 (1%) | **1.006** | 1.007 | same |
| hecbench/bitonic-sort-cuda | 0.211 (2%) | 0.211 (1%) | 0.200 (1%) | **1.054** | 1.002 | same |

geometric mean JIT / linked: **1.0360** over 13 programs (0.967 to 1.131); control: 1.0011 (0.995 to 1.011)

Fused runs on top of linking (ab-fuse.json): sides JIT, linked, linked again (the control), linked + fused, JIT + fused.

| program | JIT s | linked s | linked again s | linked + fused s | JIT + fused s | **linked / linked + fused** | control linked / linked again | JIT / linked + fused | JIT / JIT + fused | outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hecbench/black-scholes-cuda | 19.845 (1%) | 16.283 (1%) | 16.274 (1%) | 14.881 (2%) | 17.948 (2%) | **1.094** | 1.001 | 1.334 | 1.106 | same |
| rodinia/gaussian | 3.397 (1%) | 3.194 (1%) | 3.207 (3%) | 3.186 (1%) | 3.363 (3%) | **1.003** | 0.996 | 1.066 | 1.010 | same |
| rodinia/hotspot | 0.866 (1%) | 0.775 (3%) | 0.787 (48%) | 0.753 (1%) | 0.844 (2%) | **1.028** | 0.984 | 1.149 | 1.026 | same |
| polybench/SYRK | 5.007 (49%) | 4.894 (2%) | 4.910 (2%) | 4.711 (1%) | 4.773 (75%) | **1.039** | 0.997 | 1.063 | 1.049 | same |
| hecbench/bilateral-cuda | 1.093 (1%) | 0.984 (2%) | 0.982 (1%) | 0.896 (2%) | 1.013 (1%) | **1.098** | 1.002 | 1.221 | 1.080 | same |
| hecbench/bh-cuda | 0.620 (1%) | 0.593 (3%) | 0.595 (3%) | 0.558 (1%) | 0.592 (6%) | **1.063** | 0.997 | 1.111 | 1.048 | same |
| hecbench/silu-cuda | 0.393 (2%) | 0.367 (4%) | 0.367 (2%) | 0.335 (1%) | 0.360 (1%) | **1.096** | 1.001 | 1.175 | 1.093 | same |

geometric mean linked / linked + fused: **1.0596** over 7 programs (1.003 to 1.098); control 0.9968; JIT / linked + fused 1.1565

SYR2K again (ab-syr2k.json, 9 runs a side; ab-syr2k-diag.json and ab-syr2k-diag2.json, 5 runs):

* ab-syr2k.json, polybench/SYR2K: JIT 9.149 s (3%); JIT again 9.202 s (3%); linked 9.428 s (2%)
* ab-syr2k.json, polybench/SYRK: JIT 4.917 s (4%); JIT again 4.897 s (2%); linked 4.878 s (3%)
* ab-syr2k-diag.json, polybench/SYR2K: JIT 9.090 s (8%); linked 9.315 s (4%); linked, no in-code split (mutant link_nosplit) 9.475 s (1%); linked, -falign-loops=32 9.285 s (4%)
* ab-syr2k-diag2.json, polybench/SYR2K: JIT 9.155 s (3%); linked 9.366 s (2%); linked, inlining params raised 9.401 s (2%); JIT, inlining params raised 9.237 s (4%)
