| program | runs | default s | JIT cold | JIT warm | +LINK cold | +LINK warm | +LINK+FUSE cold | +LINK+FUSE warm | objects (b) | compile s (b) | same bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| polybench/2DCONV | 5 | 0.712 | 0.940 | 0.938 | | | | | 0 | 0.0 | same |
| polybench/3DCONV | 5 | 0.682 | 0.929 | 0.938 | | | | | 0 | 0.0 | same |
| polybench/3MM | 3 | 2.073 | 0.972 | 0.966 | | | | | 0 | 0.0 | same |
| polybench/ADI | 5 | 0.230 | 0.891 | 0.911 | | | | | 0 | 0.0 | same |
| polybench/ATAX | 3 | 1.444 | 0.972 | 0.988 | | | | | 0 | 0.0 | same |
| polybench/BICG | 5 | 0.280 | 0.982 | 0.994 | | | | | 0 | 0.0 | same |
| polybench/CORR | 3 | 56.574 | 1.055 | 1.054 | 1.033 | 1.050 | 1.032 | 1.057 | 1 | 1.1 | same |
| polybench/COVAR | 3 | 56.471 | 1.047 | 1.066 | 1.041 | 1.062 | 1.042 | 1.057 | 1 | 0.9 | same |
| polybench/FDTD-2D | 3 | 93.119 | 1.132 | 1.163 | 1.161 | 1.239 | 1.206 | 1.250 | 3 | 2.0 | same |
| polybench/GEMM | 5 | 0.741 | 0.978 | 0.989 | | | | | 0 | 0.0 | same |
| polybench/GEMVER | 5 | 0.644 | 0.883 | 0.934 | | | | | 0 | 0.0 | same |
| polybench/GESUMMV | 5 | 0.397 | 0.986 | 0.990 | | | | | 0 | 0.0 | same |
| polybench/JACOBI1D | 3 | 1.057 | 0.874 | 0.870 | | | | | 0 | 0.0 | same |
| polybench/JACOBI2D | 5 | 0.540 | 0.840 | 0.841 | | | | | 0 | 0.0 | same |
| polybench/LU | 3 | 28.579 | 1.169 | 1.212 | 1.224 | 1.281 | 1.211 | 1.279 | 1 | 0.6 | same |
| polybench/MVT | 3 | 1.559 | 0.979 | 0.989 | | | | | 0 | 0.0 | same |
| polybench/SYR2K | 3 | 10.337 | 1.012 | 1.177 | 1.011 | 1.169 | 1.020 | 1.200 | 1 | 1.5 | same |
| polybench/SYRK | 3 | 5.369 | 0.920 | 1.142 | 0.929 | 1.148 | 0.927 | 1.138 | 1 | 2.2 | same |
| rodinia/backprop | 5 | 0.531 | 0.921 | 0.951 | | | | | 0 | 0.0 | same |
| rodinia/gaussian | 3 | 3.639 | 0.854 | 1.029 | 0.838 | 1.150 | 0.853 | 1.112 | 1 | 0.8 | same |
| rodinia/lud | 3 | 2.947 | 0.967 | 0.969 | | | | | 0 | 0.0 | same |
| rodinia/needle | 5 | 0.331 | 1.003 | 1.007 | | | | | 0 | 0.0 | same |
| rodinia/pathfinder | 5 | 0.539 | 0.955 | 0.962 | | | | | 0 | 0.0 | DIFFERENT: fill |
| rodinia/srad_v2 | 5 | 0.028 | 1.004 | 0.989 | | | | | 0 | 0.0 | same |
| rodinia/particlefilter_float | 3 | 1.734 | 0.982 | 0.989 | | | | | 0 | 0.0 | default differs from itself |
| rodinia/sc_gpu | 5 | 0.689 | 0.955 | 0.955 | | | | | 0 | 0.0 | same |
| rodinia/hotspot | 3 | 1.009 | 0.902 | 0.897 | | | | | 0 | 0.0 | same |
| rodinia/bfs | 5 | 0.009 | 0.969 | 0.979 | | | | | 0 | 0.0 | same |
| rodinia/nn | 5 | 0.014 | 0.986 | 0.986 | | | | | 0 | 0.0 | same |
| rodinia/lavaMD | 3 | 1.834 | 1.062 | 1.058 | | | | | 0 | 0.0 | same |
| hecbench/adam-cuda | 5 | 0.024 | 0.881 | 0.918 | | | | | 0 | 0.0 | same |
| hecbench/aobench-cuda | 5 | 0.570 | 0.946 | 0.932 | | | | | 0 | 0.0 | same |
| hecbench/aop-cuda | 3 | 1.810 | 0.988 | 0.990 | | | | | 0 | 0.0 | same |
| hecbench/asmooth-cuda | 5 | 0.222 | 0.875 | 0.865 | | | | | 0 | 0.0 | same |
| hecbench/asyncAllocation-cuda | 3 | 3.285 | 0.859 | 1.291 | 0.899 | 1.520 | 0.889 | 1.589 | 1 | 2.1 | same |
| hecbench/atan2-cuda | 5 | 0.020 | 0.936 | 0.927 | | | | | 0 | 0.0 | same |
| hecbench/atomicAggregate-cuda | 3 | 3.726 | 0.934 | 1.137 | 0.928 | 1.257 | 0.929 | 1.252 | 1 | 0.7 | same |
| hecbench/atomicCost-cuda | 3 | 22.727 | 1.038 | 1.086 | 1.049 | 1.089 | 1.053 | 1.087 | 1 | 0.6 | same |
| hecbench/atomicIntrinsics-cuda | 5 | 0.181 | 0.995 | 0.998 | | | | | 0 | 0.0 | same |
| hecbench/atomicPerf-cuda | 3 | 1.062 | 0.976 | 0.974 | | | | | 0 | 0.0 | same |
| hecbench/attention-cuda | 5 | 0.023 | 0.966 | 0.978 | | | | | 0 | 0.0 | same |
| hecbench/axpby-cuda | 5 | 0.408 | 0.921 | 0.931 | | | | | 0 | 0.0 | same |
| hecbench/background-subtract-cuda | 5 | 0.003 | 0.846 | 1.031 | | | | | 0 | 0.0 | same |
| hecbench/backprop-cuda | 5 | 0.010 | 0.918 | 0.927 | | | | | 0 | 0.0 | same |
| hecbench/bgmv-cuda | 5 | 0.012 | 1.025 | 1.052 | | | | | 0 | 0.0 | same |
| hecbench/bh-cuda | 5 | 0.726 | 0.885 | 0.889 | | | | | 0 | 0.0 | same |
| hecbench/bilateral-cuda | 3 | 1.138 | 0.922 | 0.926 | | | | | 0 | 0.0 | same |
| hecbench/bitonic-sort-cuda | 5 | 0.253 | 0.919 | 0.923 | | | | | 0 | 0.0 | same |
| hecbench/black-scholes-cuda | 3 | 19.892 | 0.973 | 1.025 | 0.924 | 1.247 | 1.038 | 1.328 | 1 | 3.8 | same |
| hecbench/blockAccess-cuda | 5 | 0.014 | 0.926 | 0.907 | | | | | 0 | 0.0 | same |
| hecbench/fpdc-cuda | 3 | 176.489 | 1.185 | 1.144 | | | | | 0 | 0.0 | same (TIMEOUT) |
| hecbench/perplexity-cuda | 5 | 0.389 | 0.999 | 0.983 | | | | | 0 | 0.0 | same |
| hecbench/radixsort2-cuda | 3 | 1.996 | 0.908 | 0.907 | | | | | 0 | 0.0 | same |
| hecbench/s3d-cuda | 5 | 0.385 | 0.957 | 0.957 | | | | | 0 | 0.0 | same |
| hecbench/shmembench-cuda | 3 | 4.820 | 0.998 | 0.997 | | | | | 0 | 0.0 | same |
| hecbench/silu-cuda | 5 | 0.458 | 0.901 | 0.901 | | | | | 0 | 0.0 | same |
| hecbench/sortKV-cuda | 5 | 0.175 | 0.918 | 0.917 | | | | | 0 | 0.0 | same |
| hecbench/zerocopy-cuda | 3 | 13.765 | 0.980 | 1.042 | 1.010 | 1.079 | 1.009 | 1.080 | 1 | 0.9 | same |

| suite | programs | JIT cold | JIT warm | +LINK cold | +LINK warm | +LINK+FUSE cold | +LINK+FUSE warm |
|---|---:|---:|---:|---:|---:|---:|---:|
| polybench | 18 | 0.972 | 1.004 | 0.975 | 1.010 | 0.977 | 1.011 |
| rodinia | 12 | 0.962 | 0.980 | 0.960 | 0.989 | 0.962 | 0.986 |
| hecbench | 27 | 0.939 | 0.975 | 0.940 | 0.993 | 0.944 | 0.996 |
| all | 57 | 0.954 | 0.985 | 0.955 | 0.997 | 0.958 | 0.999 |
| programs with objects | 12 | 0.993 | 1.116 | 0.999 | 1.185 | 1.012 | 1.194 |
| programs without | 45 | 0.944 | 0.953 | 0.944 | 0.953 | 0.944 | 0.953 |

JIT cold: 39 of 58 programs more than 2 % slower than the default: polybench/JACOBI2D 0.840, hecbench/background-subtract-cuda 0.846, rodinia/gaussian 0.854, hecbench/asyncAllocation-cuda 0.859, polybench/JACOBI1D 0.874, hecbench/asmooth-cuda 0.875, hecbench/adam-cuda 0.881, polybench/GEMVER 0.883, hecbench/bh-cuda 0.885, polybench/ADI 0.891, hecbench/silu-cuda 0.901, rodinia/hotspot 0.902, hecbench/radixsort2-cuda 0.908, hecbench/backprop-cuda 0.918, hecbench/sortKV-cuda 0.918, hecbench/bitonic-sort-cuda 0.919, polybench/SYRK 0.920, hecbench/axpby-cuda 0.921, rodinia/backprop 0.921, hecbench/bilateral-cuda 0.922, hecbench/blockAccess-cuda 0.926, polybench/3DCONV 0.929, hecbench/atomicAggregate-cuda 0.934, hecbench/atan2-cuda 0.936, polybench/2DCONV 0.940, hecbench/aobench-cuda 0.946, rodinia/pathfinder 0.955, rodinia/sc_gpu 0.955, hecbench/s3d-cuda 0.957, hecbench/attention-cuda 0.966, rodinia/lud 0.967, rodinia/bfs 0.969, polybench/ATAX 0.972, polybench/3MM 0.972, hecbench/black-scholes-cuda 0.973, hecbench/atomicPerf-cuda 0.976, polybench/GEMM 0.978, polybench/MVT 0.979, hecbench/zerocopy-cuda 0.980
JIT warm: 29 of 58 programs more than 2 % slower than the default: polybench/JACOBI2D 0.841, hecbench/asmooth-cuda 0.865, polybench/JACOBI1D 0.870, hecbench/bh-cuda 0.889, rodinia/hotspot 0.897, hecbench/silu-cuda 0.901, hecbench/blockAccess-cuda 0.907, hecbench/radixsort2-cuda 0.907, polybench/ADI 0.911, hecbench/sortKV-cuda 0.917, hecbench/adam-cuda 0.918, hecbench/bitonic-sort-cuda 0.923, hecbench/bilateral-cuda 0.926, hecbench/backprop-cuda 0.927, hecbench/atan2-cuda 0.927, hecbench/axpby-cuda 0.931, hecbench/aobench-cuda 0.932, polybench/GEMVER 0.934, polybench/3DCONV 0.938, polybench/2DCONV 0.938, rodinia/backprop 0.951, rodinia/sc_gpu 0.955, hecbench/s3d-cuda 0.957, rodinia/pathfinder 0.962, polybench/3MM 0.966, rodinia/lud 0.969, hecbench/atomicPerf-cuda 0.974, hecbench/attention-cuda 0.978, rodinia/bfs 0.979
+LINK cold: 38 of 58 programs more than 2 % slower than the default: rodinia/gaussian 0.838, polybench/JACOBI2D 0.840, hecbench/background-subtract-cuda 0.846, polybench/JACOBI1D 0.874, hecbench/asmooth-cuda 0.875, hecbench/adam-cuda 0.881, polybench/GEMVER 0.883, hecbench/bh-cuda 0.885, polybench/ADI 0.891, hecbench/asyncAllocation-cuda 0.899, hecbench/silu-cuda 0.901, rodinia/hotspot 0.902, hecbench/radixsort2-cuda 0.908, hecbench/backprop-cuda 0.918, hecbench/sortKV-cuda 0.918, hecbench/bitonic-sort-cuda 0.919, hecbench/axpby-cuda 0.921, rodinia/backprop 0.921, hecbench/bilateral-cuda 0.922, hecbench/black-scholes-cuda 0.924, hecbench/blockAccess-cuda 0.926, hecbench/atomicAggregate-cuda 0.928, polybench/3DCONV 0.929, polybench/SYRK 0.929, hecbench/atan2-cuda 0.936, polybench/2DCONV 0.940, hecbench/aobench-cuda 0.946, rodinia/pathfinder 0.955, rodinia/sc_gpu 0.955, hecbench/s3d-cuda 0.957, hecbench/attention-cuda 0.966, rodinia/lud 0.967, rodinia/bfs 0.969, polybench/ATAX 0.972, polybench/3MM 0.972, hecbench/atomicPerf-cuda 0.976, polybench/GEMM 0.978, polybench/MVT 0.979
+LINK warm: 29 of 58 programs more than 2 % slower than the default: polybench/JACOBI2D 0.841, hecbench/asmooth-cuda 0.865, polybench/JACOBI1D 0.870, hecbench/bh-cuda 0.889, rodinia/hotspot 0.897, hecbench/silu-cuda 0.901, hecbench/blockAccess-cuda 0.907, hecbench/radixsort2-cuda 0.907, polybench/ADI 0.911, hecbench/sortKV-cuda 0.917, hecbench/adam-cuda 0.918, hecbench/bitonic-sort-cuda 0.923, hecbench/bilateral-cuda 0.926, hecbench/backprop-cuda 0.927, hecbench/atan2-cuda 0.927, hecbench/axpby-cuda 0.931, hecbench/aobench-cuda 0.932, polybench/GEMVER 0.934, polybench/3DCONV 0.938, polybench/2DCONV 0.938, rodinia/backprop 0.951, rodinia/sc_gpu 0.955, hecbench/s3d-cuda 0.957, rodinia/pathfinder 0.962, polybench/3MM 0.966, rodinia/lud 0.969, hecbench/atomicPerf-cuda 0.974, hecbench/attention-cuda 0.978, rodinia/bfs 0.979
+LINK+FUSE cold: 37 of 58 programs more than 2 % slower than the default: polybench/JACOBI2D 0.840, hecbench/background-subtract-cuda 0.846, rodinia/gaussian 0.853, polybench/JACOBI1D 0.874, hecbench/asmooth-cuda 0.875, hecbench/adam-cuda 0.881, polybench/GEMVER 0.883, hecbench/bh-cuda 0.885, hecbench/asyncAllocation-cuda 0.889, polybench/ADI 0.891, hecbench/silu-cuda 0.901, rodinia/hotspot 0.902, hecbench/radixsort2-cuda 0.908, hecbench/backprop-cuda 0.918, hecbench/sortKV-cuda 0.918, hecbench/bitonic-sort-cuda 0.919, hecbench/axpby-cuda 0.921, rodinia/backprop 0.921, hecbench/bilateral-cuda 0.922, hecbench/blockAccess-cuda 0.926, polybench/SYRK 0.927, polybench/3DCONV 0.929, hecbench/atomicAggregate-cuda 0.929, hecbench/atan2-cuda 0.936, polybench/2DCONV 0.940, hecbench/aobench-cuda 0.946, rodinia/pathfinder 0.955, rodinia/sc_gpu 0.955, hecbench/s3d-cuda 0.957, hecbench/attention-cuda 0.966, rodinia/lud 0.967, rodinia/bfs 0.969, polybench/ATAX 0.972, polybench/3MM 0.972, hecbench/atomicPerf-cuda 0.976, polybench/GEMM 0.978, polybench/MVT 0.979
+LINK+FUSE warm: 29 of 58 programs more than 2 % slower than the default: polybench/JACOBI2D 0.841, hecbench/asmooth-cuda 0.865, polybench/JACOBI1D 0.870, hecbench/bh-cuda 0.889, rodinia/hotspot 0.897, hecbench/silu-cuda 0.901, hecbench/blockAccess-cuda 0.907, hecbench/radixsort2-cuda 0.907, polybench/ADI 0.911, hecbench/sortKV-cuda 0.917, hecbench/adam-cuda 0.918, hecbench/bitonic-sort-cuda 0.923, hecbench/bilateral-cuda 0.926, hecbench/backprop-cuda 0.927, hecbench/atan2-cuda 0.927, hecbench/axpby-cuda 0.931, hecbench/aobench-cuda 0.932, polybench/GEMVER 0.934, polybench/3DCONV 0.938, polybench/2DCONV 0.938, rodinia/backprop 0.951, rodinia/sc_gpu 0.955, hecbench/s3d-cuda 0.957, rodinia/pathfinder 0.962, polybench/3MM 0.966, rodinia/lud 0.969, hecbench/atomicPerf-cuda 0.974, hecbench/attention-cuda 0.978, rodinia/bfs 0.979

same bits: 58 programs, 56 equal in every run of every side, 1 whose default runs differ among themselves (rodinia/particlefilter_float), 1 DIFFERENT ([('rodinia/pathfinder', ['fill'])])
not measured (over the time limit on the default): ['polybench/2MM', 'polybench/GRAMSCHM', 'hecbench/aidw-cuda', 'hecbench/aligned-types-cuda', 'hecbench/atomicReduction-cuda', 'hecbench/babelstream-cuda', 'hecbench/base64e-cuda', 'hecbench/binomial-cuda']
exit status not 0 on the default: [('hecbench/fpdc-cuda', -9)]
cold runs that left objects in the cache (side: objects after each run): {'polybench/CORR': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cc': [1, 1, 1], 'cw': [], 'dc': [1, 1, 1], 'dw': []}, 'polybench/COVAR': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cc': [1, 1, 1], 'cw': [], 'dc': [1, 1, 1], 'dw': []}, 'polybench/FDTD-2D': {'a': [], 'bc': [3, 3, 3], 'bw': [], 'cc': [3, 3, 3], 'cw': [], 'dc': [3, 3, 3], 'dw': []}, 'polybench/LU': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cc': [1, 1, 1], 'cw': [], 'dc': [1, 1, 1], 'dw': []}, 'polybench/SYR2K': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cc': [1, 1, 1], 'cw': [], 'dc': [1, 1, 1], 'dw': []}, 'polybench/SYRK': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cc': [1, 1, 1], 'cw': [], 'dc': [1, 1, 1], 'dw': []}, 'rodinia/gaussian': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cc': [1, 1, 1], 'cw': [], 'dc': [1, 1, 1], 'dw': []}, 'hecbench/atomicAggregate-cuda': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cc': [1, 1, 1], 'cw': [], 'dc': [1, 1, 1], 'dw': []}, 'hecbench/atomicCost-cuda': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cc': [1, 1, 1], 'cw': [], 'dc': [1, 1, 1], 'dw': []}, 'hecbench/black-scholes-cuda': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cc': [1, 1, 1], 'cw': [], 'dc': [1, 1, 1], 'dw': []}, 'hecbench/zerocopy-cuda': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cc': [1, 1, 1], 'cw': [], 'dc': [1, 1, 1], 'dw': []}}
