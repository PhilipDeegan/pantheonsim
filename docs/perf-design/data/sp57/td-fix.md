| program | runs | default s | JIT cold | JIT warm | +LINK warm | +LINK+FUSE warm | objects (b) | compile s (b) | same bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| polybench/2DCONV | 3 | 0.536 | 0.980 | 1.014 | | | 0 | 0.0 | same |
| polybench/ADI | 3 | 0.146 | 0.944 | 0.979 | | | 0 | 0.0 | same |
| polybench/GEMVER | 3 | 0.316 | 0.965 | 0.980 | | | 0 | 0.0 | same |
| polybench/JACOBI1D | 3 | 0.760 | 0.974 | 0.962 | | | 0 | 0.0 | same |
| polybench/JACOBI2D | 3 | 0.170 | 0.904 | 0.948 | | | 0 | 0.0 | same |
| polybench/SYR2K | 3 | 3.423 | 0.937 | 1.134 | 1.134 | 1.154 | 1 | 2.6 | same |
| polybench/SYRK | 3 | 1.781 | 0.921 | 1.104 | 1.113 | 1.122 | 1 | 2.7 | same |
| rodinia/backprop | 3 | 0.230 | 0.895 | 1.007 | | | 0 | 0.0 | same |
| rodinia/gaussian | 3 | 0.899 | 0.900 | 1.099 | 1.147 | 1.156 | 1 | 0.9 | same |
| rodinia/lud | 3 | 2.056 | 0.982 | 0.980 | | | 0 | 0.0 | same |
| rodinia/hotspot | 3 | 0.301 | 0.972 | 0.982 | | | 0 | 0.0 | same |
| hecbench/asmooth-cuda | 3 | 0.082 | 0.878 | 0.946 | | | 0 | 0.0 | same |
| hecbench/asyncAllocation-cuda | 3 | 0.996 | 0.872 | 1.242 | 1.383 | 1.422 | 1 | 2.0 | same |
| hecbench/atomicAggregate-cuda | 3 | 0.891 | 0.949 | 1.071 | 1.137 | 1.157 | 1 | 0.8 | same |
| hecbench/bh-cuda | 3 | 0.204 | 0.967 | 1.003 | | | 0 | 0.0 | default differs from itself |
| hecbench/bilateral-cuda | 3 | 0.290 | 0.980 | 0.987 | | | 0 | 0.0 | same |
| hecbench/bitonic-sort-cuda | 3 | 0.084 | 0.936 | 1.057 | | | 0 | 0.0 | same |
| hecbench/radixsort2-cuda | 3 | 0.875 | 1.000 | 1.008 | | | 0 | 0.0 | default differs from itself |

| suite | programs | JIT cold | JIT warm | +LINK warm | +LINK+FUSE warm |
|---|---:|---:|---:|---:|---:|
| polybench | 7 | 0.946 | 1.015 | 1.016 | 1.020 |
| rodinia | 4 | 0.936 | 1.016 | 1.027 | 1.029 |
| hecbench | 7 | 0.939 | 1.041 | 1.067 | 1.073 |
| all | 18 | 0.941 | 1.025 | 1.038 | 1.042 |
| programs with objects | 5 | 0.915 | 1.129 | 1.179 | 1.198 |
| programs without | 13 | 0.951 | 0.988 | 0.988 | 0.988 |

JIT cold: 14 of 18 programs more than 2 % slower than the default: hecbench/asyncAllocation-cuda 0.872, hecbench/asmooth-cuda 0.878, rodinia/backprop 0.895, rodinia/gaussian 0.900, polybench/JACOBI2D 0.904, polybench/SYRK 0.921, hecbench/bitonic-sort-cuda 0.936, polybench/SYR2K 0.937, polybench/ADI 0.944, hecbench/atomicAggregate-cuda 0.949, polybench/GEMVER 0.965, hecbench/bh-cuda 0.967, rodinia/hotspot 0.972, polybench/JACOBI1D 0.974
JIT warm: 6 of 18 programs more than 2 % slower than the default: hecbench/asmooth-cuda 0.946, polybench/JACOBI2D 0.948, polybench/JACOBI1D 0.962, polybench/ADI 0.979, rodinia/lud 0.980, polybench/GEMVER 0.980
+LINK warm: 6 of 18 programs more than 2 % slower than the default: hecbench/asmooth-cuda 0.946, polybench/JACOBI2D 0.948, polybench/JACOBI1D 0.962, polybench/ADI 0.979, rodinia/lud 0.980, polybench/GEMVER 0.980
+LINK+FUSE warm: 6 of 18 programs more than 2 % slower than the default: hecbench/asmooth-cuda 0.946, polybench/JACOBI2D 0.948, polybench/JACOBI1D 0.962, polybench/ADI 0.979, rodinia/lud 0.980, polybench/GEMVER 0.980

same bits: 18 programs, 16 equal in every run of every side, 2 whose default runs differ among themselves (hecbench/bh-cuda, hecbench/radixsort2-cuda), 0 DIFFERENT ([])
not measured (over the time limit on the default): []
exit status not 0 on the default: []
cold runs that left objects in the cache (side: objects after each run): {'polybench/SYR2K': {'a': [], 'bc': [1, 1, 1], 'bw': [], 'cw': [], 'dw': []}}
