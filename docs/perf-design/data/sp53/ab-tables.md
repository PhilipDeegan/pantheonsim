| program | JIT s (spread) | JIT again s (spread) | linked s (spread) | **JIT / linked** | control JIT / JIT again | linked + BSYNC s (spread) | linked / linked + BSYNC | JIT / linked + BSYNC | outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hecbench/black-scholes-cuda | 19.295 (2%) | 19.157 (1%) | 15.699 (1%) | **1.229** | 1.007 | 15.035 (1%) | 1.044 | 1.283 | same |
| polybench/LU | 23.638 (1%) | 23.855 (8%) | 22.361 (8%) | **1.057** | 0.991 | 22.344 (5%) | 1.001 | 1.058 | same |
| rodinia/gaussian | 3.350 (2%) | 3.348 (1%) | 3.138 (1%) | **1.068** | 1.000 | 3.132 (1%) | 1.002 | 1.069 | same |
| rodinia/hotspot | 0.841 (2%) | 0.858 (3%) | 0.744 (1%) | **1.130** | 0.981 | 0.743 (2%) | 1.001 | 1.131 | same |
| polybench/SYRK | 4.715 (2%) | 4.742 (2%) | 4.703 (3%) | **1.003** | 0.994 | (no BSYNC in it) | - | - | same |
| polybench/SYR2K | 8.862 (2%) | 8.833 (4%) | 8.910 (7%) | **0.995** | 1.003 | (no BSYNC in it) | - | - | same |
| polybench/JACOBI1D | 0.912 (8%) | 0.911 (1%) | 0.797 (1%) | **1.144** | 1.002 | 0.806 (2%) | 0.989 | 1.132 | same |
| polybench/JACOBI2D | 0.479 (1%) | 0.478 (4%) | 0.447 (2%) | **1.070** | 1.001 | 0.446 (1%) | 1.003 | 1.073 | same |
| polybench/3DCONV | 0.590 (1%) | 0.589 (1%) | 0.580 (1%) | **1.017** | 1.002 | 0.579 (2%) | 1.001 | 1.018 | same |
| polybench/2DCONV | 0.651 (3%) | 0.652 (2%) | 0.640 (3%) | **1.018** | 0.999 | 0.638 (2%) | 1.003 | 1.021 | same |
| rodinia/lud | 2.895 (1%) | 2.902 (0%) | 2.890 (0%) | **1.002** | 0.998 | 2.917 (1%) | 0.991 | 0.992 | same |
| rodinia/backprop | 0.507 (27%) | 0.505 (0%) | 0.507 (1%) | **1.001** | 1.005 | 0.511 (1%) | 0.991 | 0.992 | same |
| rodinia/pathfinder | 0.513 (4%) | 0.520 (3%) | 0.520 (4%) | **0.987** | 0.988 | 0.525 (3%) | 0.991 | 0.979 | same |
| hecbench/bilateral-cuda | 1.030 (1%) | 1.036 (6%) | 0.913 (1%) | **1.128** | 0.994 | 0.918 (0%) | 0.995 | 1.122 | same |
| hecbench/bh-cuda | 0.593 (2%) | 0.592 (1%) | 0.573 (30%) | **1.035** | 1.002 | 0.545 (1%) | 1.051 | 1.088 | same |
| hecbench/silu-cuda | 0.364 (4%) | 0.366 (2%) | 0.338 (1%) | **1.077** | 0.996 | 0.315 (2%) | 1.073 | 1.156 | same |
| hecbench/aop-cuda | 1.797 (1%) | 1.801 (1%) | 1.804 (2%) | **0.996** | 0.998 | 1.798 (2%) | 1.003 | 1.000 | same |
| hecbench/bitonic-sort-cuda | 0.202 (1%) | 0.201 (2%) | 0.189 (3%) | **1.065** | 1.002 | 0.200 (1%) | 0.945 | 1.006 | same |

geometric mean JIT / linked: **1.0549** over 18 programs (0.987 to 1.229); control: 0.9979 (0.981 to 1.007); linked / linked + BSYNC over 16 programs: 1.0048 (0.945 to 1.073); JIT / linked + BSYNC: 1.0673

| program | JIT s (spread) | JIT again s (spread) | linked (left unlinked by the rule) s (spread) | linked, `VGPU_JIT_LINK_RULE=0` s (spread) | JIT / linked | JIT / linked, rule off | control |
|---|---:|---:|---:|---:|---:|---:|---:|
| polybench/SYR2K | 8.862 (2%) | 8.833 (4%) | 8.910 (7%) | 8.876 (3%) | 0.995 | 0.998 | 1.003 |
| polybench/SYRK | 4.715 (2%) | 4.742 (2%) | 4.703 (3%) | 4.706 (2%) | 1.003 | 1.002 | 0.994 |
