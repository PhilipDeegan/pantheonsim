| program | region calls M | warp instructions a call | split-branch | uniform-branch-taken | uniform-branch-not-taken | register-only-op | store-gate | to-compiled-region | bssy-nop | bsync | exit | exit-no-lane-leaves | bar-yield-call-ret-warpsync | straight-op-outside-regions | interpreter-op | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hecbench/black-scholes-cuda | 115.61 | 7.9 | 5.1 | 14.1 | 19.2 | 2.6 | 1.3 | 3.8 | 16.7 | 21.8 | 0.0 | 1.3 | 14.1 | 0.0 | 0.0 | 0.0 |
| polybench/LU | 136.62 | 11.7 | 0.0 | 0.0 | 0.0 | 0.0 | 33.5 | 0.0 | 0.0 | 0.0 | 33.6 | 32.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| rodinia/gaussian | 15.45 | 9.5 | 0.0 | 0.0 | 0.0 | 0.0 | 18.2 | 0.0 | 0.0 | 0.0 | 36.7 | 45.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| rodinia/hotspot | 3.35 | 14.1 | 0.0 | 0.0 | 0.0 | 0.0 | 5.7 | 25.7 | 17.1 | 0.0 | 0.0 | 0.0 | 34.3 | 8.6 | 8.6 | 0.0 |
| polybench/SYRK | 0.10 | 1976.6 | 0.0 | 0.0 | 0.0 | 0.0 | 33.3 | 0.0 | 0.0 | 0.0 | 33.3 | 33.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| polybench/SYR2K | 0.10 | 4062.0 | 0.0 | 0.0 | 0.0 | 0.0 | 33.3 | 0.0 | 0.0 | 0.0 | 33.3 | 33.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| polybench/JACOBI1D | 3.82 | 7.0 | 0.0 | 0.0 | 0.0 | 0.0 | 33.3 | 0.0 | 0.0 | 0.0 | 1.0 | 65.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| polybench/JACOBI2D | 2.54 | 11.8 | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 | 0.0 | 0.0 | 0.0 | 3.2 | 46.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| polybench/3DCONV | 1.03 | 26.5 | 0.0 | 0.0 | 0.0 | 0.0 | 49.8 | 0.0 | 0.0 | 0.0 | 12.8 | 37.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| polybench/2DCONV | 1.04 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 | 0.0 | 0.0 | 0.0 | 0.8 | 49.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| rodinia/lud | 0.78 | 25.4 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 | 0.0 | 4.1 | 4.1 | 0.3 | 0.0 | 91.2 | 0.0 | 0.0 | 0.0 |
| rodinia/backprop | 0.65 | 19.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20.0 | 0.0 | 80.0 | 0.0 | 0.0 | 0.0 |
| rodinia/pathfinder | 0.76 | 11.3 | 0.0 | 0.0 | 0.0 | 0.0 | 2.4 | 0.0 | 0.0 | 2.4 | 0.0 | 0.0 | 92.8 | 2.4 | 0.0 | 0.0 |
| hecbench/bilateral-cuda | 4.57 | 14.2 | 1.5 | 3.0 | 4.5 | 3.0 | 0.0 | 0.0 | 34.4 | 25.0 | 0.0 | 0.0 | 25.9 | 2.6 | 0.0 | 0.0 |
| hecbench/bh-cuda | 4.14 | 9.8 | 0.2 | 0.0 | 0.0 | 0.3 | 0.0 | 15.1 | 35.0 | 35.5 | 0.0 | 0.0 | 0.2 | 0.0 | 13.3 | 0.2 |
| hecbench/silu-cuda | 2.06 | 11.7 | 0.0 | 0.0 | 0.0 | 0.0 | 8.7 | 7.6 | 31.1 | 23.3 | 1.9 | 11.7 | 15.6 | 0.0 | 0.0 | 0.0 |
| hecbench/aop-cuda | 0.28 | 22.2 | 0.0 | 0.0 | 0.0 | 0.0 | 31.8 | 0.0 | 0.0 | 4.5 | 0.0 | 31.8 | 0.0 | 31.8 | 0.0 | 0.0 |
| hecbench/bitonic-sort-cuda | 0.49 | 32.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 43.0 | 0.0 | 43.0 | 13.9 | 0.0 | 0.0 | 0.0 | 0.0 |

| program | calls of generated code M (with those that ran nothing) | ran nothing % | in a diverged warp % | of the diverged ones: split / uniform branch / register-only op / other % |
|---|---:|---:|---:|---|
| hecbench/black-scholes-cuda | 121.86 | 5.1 | 62.8 | 2 / 53 / 4 / 41 |
| polybench/LU | 136.62 | 0.0 | 0.0 | - / - / - / - |
| rodinia/gaussian | 15.45 | 0.0 | 0.0 | - / - / - / - |
| rodinia/hotspot | 3.35 | 0.0 | 0.0 | - / - / - / - |
| polybench/SYRK | 0.10 | 0.0 | 0.0 | - / - / - / - |
| polybench/SYR2K | 0.10 | 0.0 | 0.0 | - / - / - / - |
| polybench/JACOBI1D | 3.82 | 0.0 | 0.0 | - / - / - / - |
| polybench/JACOBI2D | 2.54 | 0.0 | 0.0 | - / - / - / - |
| polybench/3DCONV | 1.03 | 0.0 | 0.0 | - / - / - / - |
| polybench/2DCONV | 1.04 | 0.0 | 0.0 | - / - / - / - |
| rodinia/lud | 0.78 | 0.0 | 0.6 | 0 / 50 / 0 / 50 |
| rodinia/backprop | 0.65 | 0.0 | 0.0 | - / - / - / - |
| rodinia/pathfinder | 0.76 | 0.0 | 0.0 | 0 / 0 / 0 / 100 |
| hecbench/bilateral-cuda | 4.71 | 3.0 | 13.4 | 0 / 56 / 22 / 22 |
| hecbench/bh-cuda | 4.14 | 0.0 | 0.8 | 9 / 11 / 34 / 46 |
| hecbench/silu-cuda | 2.06 | 0.0 | 0.0 | - / - / - / - |
| hecbench/aop-cuda | 0.28 | 0.0 | 0.0 | - / - / - / - |
| hecbench/bitonic-sort-cuda | 0.49 | 0.0 | 0.0 | - / - / - / - |

| program | region calls M: unlinked | linked | ratio | warp instructions a call: unlinked | linked | continued inside generated code M: unlinked | linked | of them linked in a diverged warp M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hecbench/black-scholes-cuda | 115.61 | 67.18 | 1.72 | 7.9 | 16.7 | 35.93 | 149.99 | 37.50 |
| polybench/LU | 136.62 | 91.82 | 1.49 | 11.7 | 17.9 | 408.72 | 453.68 | 0.00 |
| rodinia/gaussian | 15.45 | 8.49 | 1.82 | 9.5 | 18.1 | 44.75 | 51.72 | 0.00 |
| rodinia/hotspot | 3.35 | 1.92 | 1.75 | 14.1 | 33.8 | 6.04 | 9.20 | 0.00 |
| polybench/SYRK | 0.10 | 0.07 | 1.50 | 1976.6 | 2964.8 | 2.55 | 2.59 | 0.00 |
| polybench/SYR2K | 0.10 | 0.07 | 1.50 | 4062.0 | 6093.1 | 4.36 | 4.39 | 0.00 |
| polybench/JACOBI1D | 3.82 | 2.59 | 1.48 | 7.0 | 17.2 | 8.92 | 12.71 | 0.00 |
| polybench/JACOBI2D | 2.54 | 1.35 | 1.88 | 11.8 | 23.0 | 7.63 | 8.82 | 0.00 |
| polybench/3DCONV | 1.03 | 0.65 | 1.60 | 26.5 | 42.8 | 2.59 | 2.97 | 0.00 |
| polybench/2DCONV | 1.04 | 0.53 | 1.97 | 20.0 | 40.3 | 3.65 | 4.17 | 0.00 |
| rodinia/lud | 0.78 | 0.72 | 1.09 | 25.4 | 27.4 | 3.50 | 3.54 | 0.00 |
| rodinia/backprop | 0.65 | 0.52 | 1.25 | 19.4 | 23.2 | 0.65 | 0.65 | 0.00 |
| rodinia/pathfinder | 0.76 | 0.74 | 1.02 | 11.3 | 11.6 | 0.47 | 0.53 | 0.00 |
| hecbench/bilateral-cuda | 4.57 | 2.45 | 1.87 | 14.2 | 32.2 | 3.99 | 10.49 | 0.14 |
| hecbench/bh-cuda | 4.14 | 2.05 | 2.02 | 9.8 | 20.7 | 3.79 | 5.57 | 0.01 |
| hecbench/silu-cuda | 2.06 | 1.11 | 1.87 | 11.7 | 24.8 | 2.09 | 3.90 | 0.00 |
| hecbench/aop-cuda | 0.28 | 0.19 | 1.46 | 22.2 | 37.3 | 0.55 | 0.74 | 0.00 |
| hecbench/bitonic-sort-cuda | 0.49 | 0.42 | 1.16 | 32.8 | 40.3 | 0.76 | 1.26 | 0.00 |
