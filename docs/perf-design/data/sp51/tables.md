### Split, SASS engine (the default), verdict file warm: percent of samples

| program | wall ms | samples | loader | device-init | module-load | sass-decode | tier-build | ptx-parse | execute | teardown | sim-other | program-host | **load+decode+compile** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| samples/simpleTemplates | 3.5 | 513 | 62.0 | 3.1 | 0.4 | 1.9 | 0.2 | 0.0 | 4.1 | 1.2 | 18.3 | 8.8 | **2.5** |
| rodinia/srad_v2 | 4.5 | 737 | 52.5 | 2.7 | 0.0 | 6.8 | 0.5 | 0.1 | 4.6 | 0.3 | 19.1 | 13.3 | **7.5** |
| samples/cdpSimpleQuicksort | 4.5 | 883 | 37.4 | 3.1 | 4.8 | 7.2 | 0.3 | 0.3 | 7.4 | 0.2 | 20.8 | 18.5 | **12.7** |
| samples/vectorAdd | 4.5 | 820 | 45.1 | 3.7 | 0.2 | 1.3 | 0.0 | 0.0 | 13.9 | 0.0 | 11.3 | 24.4 | **1.6** |
| samples/cdpQuadtree | 8.1 | 1647 | 25.0 | 1.5 | 3.9 | 8.8 | 0.8 | 0.1 | 31.3 | 0.2 | 14.7 | 13.8 | **13.7** |
| rodinia/bfs | 8.3 | 1532 | 20.8 | 1.8 | 0.3 | 1.1 | 0.3 | 0.2 | 46.9 | 0.3 | 6.1 | 22.3 | **1.9** |
| hecbench/backprop-cuda | 9.7 | 1896 | 16.9 | 1.9 | 0.3 | 1.0 | 0.1 | 0.2 | 50.8 | 0.3 | 6.5 | 22.0 | **1.6** |
| samples/binaryPartitionCG | 10.7 | 2209 | 15.3 | 1.1 | 0.1 | 1.6 | 0.0 | 0.0 | 64.6 | 0.3 | 8.5 | 8.3 | **1.9** |
| rodinia/gaussian | 12.5 | 2176 | 15.2 | 1.5 | 0.4 | 1.2 | 0.1 | 0.1 | 65.8 | 0.2 | 6.3 | 9.2 | **1.8** |
| hecbench/atan2-cuda | 17.7 | 3560 | 7.5 | 0.7 | 0.0 | 1.8 | 0.1 | 0.0 | 61.9 | 0.0 | 4.4 | 23.5 | **2.0** |
| samples/fp16ScalarProduct | 19.0 | 3784 | 9.6 | 0.6 | 0.1 | 0.5 | 0.1 | 0.0 | 29.6 | 0.1 | 5.0 | 54.5 | **0.7** |
| hecbench/adam-cuda | 22.0 | 4402 | 6.7 | 0.7 | 0.0 | 1.5 | 0.2 | 0.0 | 76.1 | 0.0 | 2.7 | 12.0 | **1.7** |
| samples/scalarProd | 31.7 | 6251 | 5.5 | 0.4 | 0.1 | 0.8 | 0.0 | 0.0 | 39.4 | 0.1 | 5.7 | 47.9 | **1.0** |
| rodinia/lud | 33.3 | 6475 | 6.1 | 0.3 | 0.0 | 2.4 | 0.2 | 0.0 | 69.9 | 0.0 | 4.2 | 16.8 | **2.7** |
| rodinia/backprop | 39.4 | 8187 | 5.1 | 0.4 | 0.0 | 0.6 | 0.0 | 0.0 | 71.2 | 0.1 | 3.5 | 19.1 | **0.7** |
| rodinia/particlefilter_float | 80.6 | 16769 | 2.1 | 0.2 | 0.0 | 1.1 | 0.2 | 0.0 | 93.5 | 0.0 | 1.6 | 1.3 | **1.4** |
| samples/shfl_scan | 94.3 | 19240 | 1.7 | 0.2 | 0.0 | 0.5 | 0.0 | 0.0 | 81.4 | 0.0 | 7.3 | 8.8 | **0.6** |
| hecbench/sortKV-cuda | 168.2 | 34790 | 1.0 | 0.1 | 0.1 | 2.0 | 0.3 | 0.0 | 80.3 | 0.0 | 4.0 | 12.2 | **2.4** |
| polybench/ADI | 215.0 | 49182 | 0.8 | 0.2 | 0.0 | 0.4 | 0.0 | 0.1 | 75.7 | 0.0 | 1.6 | 21.2 | **0.5** |
| polybench/BICG | 303.4 | 63785 | 0.6 | 0.2 | 0.0 | 0.1 | 0.0 | 0.0 | 62.3 | 0.0 | 3.4 | 33.4 | **0.1** |
| mean of shares | | | 16.8 | 1.2 | 0.5 | 2.1 | 0.2 | 0.1 | 51.5 | 0.2 | 7.8 | 19.6 | **2.9** |
| wall-weighted | 1090.9 | | 3.3 | 0.3 | 0.1 | 0.9 | 0.1 | 0.0 | 70.0 | 0.0 | 3.9 | 21.3 | **1.1** |

Ceiling (every sample of those four columns removed, nothing added): geomean 1.031x over 20 programs; 0 of 20 programs at or above 15 %.

### Split, PTX engine (`VGPU_SASS=0`): percent of samples

| program | wall ms | samples | loader | device-init | module-load | sass-decode | tier-build | ptx-parse | execute | teardown | sim-other | program-host | **load+decode+compile** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| samples/simpleTemplates | 2.8 | 584 | 57.9 | 3.8 | 5.8 | 0.0 | 0.0 | 4.8 | 3.9 | 0.9 | 17.8 | 5.1 | **10.6** |
| samples/vectorAdd | 4.6 | 890 | 33.6 | 2.4 | 3.8 | 0.0 | 0.0 | 2.4 | 26.2 | 0.3 | 9.2 | 22.1 | **6.2** |
| samples/cdpSimpleQuicksort | 5.0 | 1053 | 37.0 | 2.6 | 11.5 | 0.0 | 0.0 | 9.3 | 7.6 | 0.2 | 16.2 | 15.6 | **20.8** |
| rodinia/srad_v2 | 5.5 | 989 | 36.4 | 2.8 | 2.2 | 0.0 | 0.0 | 16.8 | 3.0 | 0.2 | 18.7 | 19.8 | **19.0** |
| rodinia/bfs | 12.4 | 2382 | 14.0 | 1.2 | 1.0 | 0.0 | 0.0 | 2.2 | 62.6 | 0.1 | 5.5 | 13.4 | **3.2** |
| samples/binaryPartitionCG | 14.3 | 2884 | 10.9 | 1.1 | 2.3 | 0.0 | 0.0 | 4.5 | 61.0 | 0.7 | 11.3 | 8.3 | **6.8** |
| hecbench/backprop-cuda | 15.4 | 3166 | 12.6 | 1.0 | 0.9 | 0.0 | 0.0 | 1.9 | 66.3 | 0.1 | 5.1 | 12.1 | **2.7** |
| samples/cdpQuadtree | 16.7 | 3446 | 12.0 | 0.8 | 5.8 | 0.0 | 0.0 | 22.8 | 22.2 | 0.1 | 15.4 | 20.9 | **28.6** |
| rodinia/gaussian | 24.2 | 4846 | 7.7 | 0.5 | 0.5 | 0.0 | 0.0 | 1.1 | 84.5 | 0.1 | 3.5 | 2.1 | **1.6** |
| hecbench/atan2-cuda | 28.6 | 5805 | 7.0 | 0.5 | 0.4 | 0.0 | 0.0 | 1.7 | 70.7 | 0.1 | 3.3 | 16.3 | **2.1** |
| hecbench/adam-cuda | 32.1 | 6613 | 5.1 | 0.4 | 0.5 | 0.0 | 0.0 | 2.9 | 76.4 | 0.1 | 3.3 | 11.3 | **3.4** |
| samples/fp16ScalarProduct | 33.2 | 6701 | 5.5 | 0.5 | 1.0 | 0.0 | 0.0 | 1.9 | 55.4 | 0.1 | 3.5 | 32.0 | **2.9** |
| samples/scalarProd | 33.6 | 6822 | 5.5 | 0.6 | 0.9 | 0.0 | 0.0 | 1.7 | 41.5 | 0.1 | 5.6 | 44.2 | **2.6** |
| rodinia/backprop | 65.4 | 13431 | 2.8 | 0.2 | 0.2 | 0.0 | 0.0 | 0.8 | 82.1 | 0.0 | 2.7 | 11.1 | **1.0** |
| rodinia/lud | 86.8 | 18325 | 2.3 | 0.1 | 0.2 | 0.0 | 0.0 | 2.3 | 86.4 | 0.0 | 2.0 | 6.7 | **2.5** |
| samples/shfl_scan | 165.1 | 34265 | 1.2 | 0.1 | 0.4 | 0.0 | 0.0 | 1.5 | 88.7 | 0.0 | 2.6 | 5.6 | **1.8** |
| rodinia/particlefilter_float | 176.1 | 36551 | 1.0 | 0.1 | 0.2 | 0.0 | 0.0 | 2.2 | 91.0 | 0.0 | 2.7 | 3.0 | **2.3** |
| polybench/ADI | 186.3 | 38793 | 0.9 | 0.2 | 0.1 | 0.0 | 0.0 | 0.5 | 72.2 | 0.0 | 2.2 | 23.9 | **0.6** |
| polybench/BICG | 444.6 | 93954 | 0.4 | 0.1 | 0.0 | 0.0 | 0.0 | 0.1 | 76.4 | 0.0 | 2.2 | 20.7 | **0.1** |
| hecbench/sortKV-cuda | 765.5 | 163058 | 0.2 | 0.0 | 0.2 | 0.0 | 0.0 | 3.0 | 80.6 | 0.0 | 8.2 | 7.8 | **3.2** |
| mean of shares | | | 12.7 | 0.9 | 1.9 | 0.0 | 0.0 | 4.2 | 57.9 | 0.2 | 7.1 | 15.1 | **6.1** |
| wall-weighted | 2118.3 | | 1.7 | 0.2 | 0.3 | 0.0 | 0.0 | 2.0 | 78.2 | 0.0 | 4.9 | 12.7 | **2.3** |

Ceiling (every sample of those four columns removed, nothing added): geomean 1.069x over 20 programs; 3 of 20 programs at or above 15 %.

### The verdict file there already is (`VGPU_SASS_CACHE`): off, cold, warm

| program | off ms | cold ms | warm ms | off/warm | off/cold | outputs |
|---|---:|---:|---:|---:|---:|---|
| samples/simpleTemplates | 2.92 | 3.41 | 2.69 | 1.087 | 0.856 | same |
| samples/vectorAdd | 3.90 | 4.72 | 3.79 | 1.030 | 0.827 | same |
| rodinia/srad_v2 | 4.33 | 4.19 | 4.30 | 1.008 | 1.034 | same |
| samples/cdpSimpleQuicksort | 4.92 | 5.92 | 4.50 | 1.092 | 0.830 | same |
| samples/cdpQuadtree | 8.93 | 9.41 | 8.25 | 1.082 | 0.948 | same |
| rodinia/bfs | 8.63 | 8.08 | 8.43 | 1.025 | 1.068 | same |
| hecbench/backprop-cuda | 9.24 | 9.93 | 9.46 | 0.976 | 0.930 | same |
| samples/binaryPartitionCG | 11.10 | 11.41 | 10.93 | 1.016 | 0.973 | same |
| rodinia/gaussian | 11.08 | 11.09 | 10.96 | 1.011 | 0.999 | same |
| hecbench/atan2-cuda | 18.32 | 18.90 | 18.02 | 1.016 | 0.969 | same |
| samples/fp16ScalarProduct | 19.58 | 19.47 | 19.34 | 1.012 | 1.005 | same |
| hecbench/adam-cuda | 22.40 | 22.62 | 22.16 | 1.011 | 0.990 | same |
| rodinia/lud | 32.53 | 32.25 | 31.79 | 1.023 | 1.009 | same |
| samples/scalarProd | 32.25 | 32.58 | 32.45 | 0.994 | 0.990 | same |
| rodinia/backprop | 35.41 | 35.55 | 35.45 | 0.999 | 0.996 | same |
| rodinia/particlefilter_float | 82.52 | 81.91 | 81.55 | 1.012 | 1.007 | DIFFER |
| samples/shfl_scan | 95.38 | 95.85 | 96.12 | 0.992 | 0.995 | same |
| hecbench/sortKV-cuda | 173.47 | 174.55 | 170.69 | 1.016 | 0.994 | same |
| polybench/ADI | 216.92 | 215.41 | 216.01 | 1.004 | 1.007 | same |
| polybench/BICG | 277.14 | 277.36 | 276.99 | 1.001 | 0.999 | same |

geomean off/warm = 1.0199 over 20 programs

geomean off/cold = 0.9692 over 20 programs

