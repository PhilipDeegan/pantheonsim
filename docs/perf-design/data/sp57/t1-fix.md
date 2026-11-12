| program | runs | default s | JIT cold | JIT warm | objects (b) | compile s (b) | same bits |
|---|---:|---:|---:|---:|---:|---:|---|
| polybench/2DCONV | 5 | 0.713 | 0.990 | 0.998 | 0 | 0.0 | same |
| polybench/3DCONV | 5 | 0.692 | 0.996 | 1.004 | 0 | 0.0 | same |
| polybench/3MM | 3 | 2.067 | 1.002 | 0.997 | 0 | 0.0 | same |
| polybench/ADI | 5 | 0.231 | 0.935 | 0.961 | 0 | 0.0 | same |
| polybench/ATAX | 3 | 1.530 | 0.995 | 0.992 | 0 | 0.0 | same |
| polybench/BICG | 5 | 0.296 | 0.980 | 0.992 | 0 | 0.0 | same |
| polybench/GEMM | 5 | 0.743 | 1.000 | 0.989 | 0 | 0.0 | same |
| polybench/GEMVER | 5 | 0.631 | 0.972 | 0.992 | 0 | 0.0 | same |
| polybench/GESUMMV | 5 | 0.396 | 0.995 | 0.988 | 0 | 0.0 | same |
| polybench/JACOBI1D | 3 | 1.052 | 0.991 | 0.982 | 0 | 0.0 | same |
| polybench/JACOBI2D | 5 | 0.548 | 0.989 | 0.970 | 0 | 0.0 | same |
| polybench/LU | 3 | 29.144 | 1.182 | 1.224 | 1 | 0.6 | same |
| polybench/MVT | 3 | 1.547 | 0.986 | 1.000 | 0 | 0.0 | same |
| polybench/SYR2K | 3 | 11.721 | 1.143 | 1.321 | 1 | 1.4 | same |
| polybench/SYRK | 3 | 5.507 | 0.946 | 1.164 | 1 | 2.3 | same |
| rodinia/backprop | 5 | 0.531 | 0.965 | 0.996 | 0 | 0.0 | same |
| rodinia/gaussian | 3 | 3.821 | 0.994 | 1.120 | 1 | 0.8 | same |
| rodinia/lud | 3 | 2.958 | 0.995 | 0.997 | 0 | 0.0 | same |
| rodinia/needle | 5 | 0.335 | 1.006 | 1.016 | 0 | 0.0 | same |
| rodinia/pathfinder | 5 | 0.539 | 0.981 | 0.999 | 0 | 0.0 | DIFFERENT: fill |
| rodinia/srad_v2 | 5 | 0.027 | 0.986 | 0.975 | 0 | 0.0 | same |
| rodinia/particlefilter_float | 3 | 1.723 | 0.998 | 0.991 | 0 | 0.0 | default differs from itself |
| rodinia/sc_gpu | 5 | 0.687 | 0.996 | 0.990 | 0 | 0.0 | same |
| rodinia/hotspot | 3 | 0.996 | 0.984 | 0.988 | 0 | 0.0 | same |
| rodinia/bfs | 5 | 0.009 | 0.989 | 0.989 | 0 | 0.0 | same |
| rodinia/nn | 5 | 0.014 | 1.037 | 1.037 | 0 | 0.0 | same |
| rodinia/lavaMD | 3 | 1.730 | 0.997 | 0.986 | 0 | 0.0 | same |
| hecbench/adam-cuda | 5 | 0.024 | 0.912 | 0.930 | 0 | 0.0 | same |
| hecbench/aobench-cuda | 5 | 0.570 | 1.002 | 1.000 | 0 | 0.0 | same |
| hecbench/aop-cuda | 3 | 1.820 | 1.002 | 1.003 | 0 | 0.0 | same |
| hecbench/asmooth-cuda | 5 | 0.223 | 0.934 | 0.956 | 0 | 0.0 | same |
| hecbench/asyncAllocation-cuda | 3 | 3.153 | 0.870 | 1.270 | 1 | 2.1 | same |
| hecbench/atan2-cuda | 5 | 0.020 | 0.891 | 0.900 | 0 | 0.0 | same |
| hecbench/atomicAggregate-cuda | 3 | 3.684 | 0.981 | 1.126 | 1 | 0.7 | same |
| hecbench/atomicCost-cuda | 3 | 22.565 | 1.052 | 1.071 | 1 | 0.6 | same |
| hecbench/atomicIntrinsics-cuda | 5 | 0.183 | 1.004 | 0.998 | 0 | 0.0 | same |
| hecbench/atomicPerf-cuda | 3 | 1.062 | 0.968 | 0.972 | 0 | 0.0 | same |
| hecbench/attention-cuda | 5 | 0.022 | 0.965 | 0.941 | 0 | 0.0 | same |
| hecbench/axpby-cuda | 5 | 0.404 | 0.997 | 0.997 | 0 | 0.0 | same |
| hecbench/background-subtract-cuda | 5 | 0.004 | 0.975 | 1.182 | 0 | 0.0 | same |
| hecbench/backprop-cuda | 5 | 0.011 | 0.917 | 0.982 | 0 | 0.0 | same |
| hecbench/bgmv-cuda | 5 | 0.011 | 0.933 | 0.888 | 0 | 0.0 | same |
| hecbench/bh-cuda | 5 | 0.737 | 1.004 | 1.000 | 0 | 0.0 | same |
| hecbench/bilateral-cuda | 3 | 1.132 | 0.997 | 0.983 | 0 | 0.0 | same |
| hecbench/bitonic-sort-cuda | 5 | 0.261 | 0.992 | 0.987 | 0 | 0.0 | same |
| hecbench/black-scholes-cuda | 3 | 19.741 | 0.980 | 1.025 | 1 | 4.0 | same |
| hecbench/blockAccess-cuda | 5 | 0.013 | 0.978 | 0.944 | 0 | 0.0 | same |
| hecbench/perplexity-cuda | 5 | 0.376 | 0.994 | 1.008 | 0 | 0.0 | same |
| hecbench/radixsort2-cuda | 3 | 1.992 | 0.987 | 0.991 | 0 | 0.0 | same |
| hecbench/s3d-cuda | 5 | 0.389 | 0.956 | 0.961 | 0 | 0.0 | same |
| hecbench/shmembench-cuda | 3 | 4.880 | 1.005 | 1.002 | 0 | 0.0 | same |
| hecbench/silu-cuda | 5 | 0.465 | 0.963 | 0.978 | 0 | 0.0 | same |
| hecbench/sortKV-cuda | 5 | 0.173 | 0.924 | 0.921 | 0 | 0.0 | same |
| hecbench/zerocopy-cuda | 3 | 13.900 | 1.012 | 1.041 | 1 | 0.6 | same |

| suite | programs | JIT cold | JIT warm |
|---|---:|---:|---:|
| polybench | 15 | 1.005 | 1.034 |
| rodinia | 12 | 0.994 | 1.006 |
| hecbench | 27 | 0.969 | 0.999 |
| all | 54 | 0.985 | 1.010 |
| programs with objects | 9 | 1.014 | 1.147 |
| programs without | 45 | 0.979 | 0.985 |

JIT cold: 18 of 54 programs more than 2 % slower than the default: hecbench/asyncAllocation-cuda 0.870, hecbench/atan2-cuda 0.891, hecbench/adam-cuda 0.912, hecbench/backprop-cuda 0.917, hecbench/sortKV-cuda 0.924, hecbench/bgmv-cuda 0.933, hecbench/asmooth-cuda 0.934, polybench/ADI 0.935, polybench/SYRK 0.946, hecbench/s3d-cuda 0.956, hecbench/silu-cuda 0.963, hecbench/attention-cuda 0.965, rodinia/backprop 0.965, hecbench/atomicPerf-cuda 0.968, polybench/GEMVER 0.972, hecbench/background-subtract-cuda 0.975, hecbench/blockAccess-cuda 0.978, hecbench/black-scholes-cuda 0.980
JIT warm: 13 of 54 programs more than 2 % slower than the default: hecbench/bgmv-cuda 0.888, hecbench/atan2-cuda 0.900, hecbench/sortKV-cuda 0.921, hecbench/adam-cuda 0.930, hecbench/attention-cuda 0.941, hecbench/blockAccess-cuda 0.944, hecbench/asmooth-cuda 0.956, polybench/ADI 0.961, hecbench/s3d-cuda 0.961, polybench/JACOBI2D 0.970, hecbench/atomicPerf-cuda 0.972, rodinia/srad_v2 0.975, hecbench/silu-cuda 0.978

same bits: 54 programs, 52 equal in every run of every side, 1 whose default runs differ among themselves (rodinia/particlefilter_float), 1 DIFFERENT ([('rodinia/pathfinder', ['fill'])])
not measured (over the time limit on the default): []
exit status not 0 on the default: []
cold runs that left objects in the cache (side: objects after each run): {'polybench/LU': {'a': [], 'bc': [1, 1, 1], 'bw': []}, 'polybench/SYR2K': {'a': [], 'bc': [1, 1, 1], 'bw': []}, 'polybench/SYRK': {'a': [], 'bc': [1, 1, 1], 'bw': []}, 'rodinia/gaussian': {'a': [], 'bc': [1, 1, 1], 'bw': []}, 'hecbench/atomicAggregate-cuda': {'a': [], 'bc': [1, 1, 1], 'bw': []}, 'hecbench/atomicCost-cuda': {'a': [], 'bc': [1, 1, 1], 'bw': []}, 'hecbench/black-scholes-cuda': {'a': [], 'bc': [1, 1, 1], 'bw': []}, 'hecbench/zerocopy-cuda': {'a': [], 'bc': [1, 1, 1], 'bw': []}}
