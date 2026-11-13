| program | runs | default s | JIT cold | JIT warm | +LINK warm | +LINK+FUSE warm | objects (b) | compile s (b) | same bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| polybench/2DCONV | 3 | 0.526 | 0.979 | 0.994 | | | 0 | 0.0 | same |
| polybench/ADI | 3 | 0.148 | 0.951 | 0.984 | | | 0 | 0.0 | same |
| polybench/GEMVER | 3 | 0.301 | 0.869 | 0.944 | | | 0 | 0.0 | same |
| polybench/JACOBI1D | 3 | 0.752 | 0.914 | 0.903 | | | 0 | 0.0 | same |
| polybench/JACOBI2D | 3 | 0.171 | 0.869 | 0.904 | | | 0 | 0.0 | same |
| polybench/SYR2K | 3 | 3.437 | 0.931 | 1.138 | 1.140 | 1.148 | 1 | 3.0 | same |

| suite | programs | JIT cold | JIT warm | +LINK warm | +LINK+FUSE warm |
|---|---:|---:|---:|---:|---:|
| polybench | 6 | 0.918 | 0.975 | 0.975 | 0.976 |
| rodinia | 0 | nan | nan | nan | nan |
| hecbench | 0 | nan | nan | nan | nan |
| all | 6 | 0.918 | 0.975 | 0.975 | 0.976 |
| programs with objects | 1 | 0.931 | 1.138 | 1.140 | 1.148 |
| programs without | 5 | 0.915 | 0.945 | 0.945 | 0.945 |

JIT cold: 6 of 6 programs more than 2 % slower than the default: polybench/GEMVER 0.869, polybench/JACOBI2D 0.869, polybench/JACOBI1D 0.914, polybench/SYR2K 0.931, polybench/ADI 0.951, polybench/2DCONV 0.979
JIT warm: 3 of 6 programs more than 2 % slower than the default: polybench/JACOBI1D 0.903, polybench/JACOBI2D 0.904, polybench/GEMVER 0.944
+LINK warm: 3 of 6 programs more than 2 % slower than the default: polybench/JACOBI1D 0.903, polybench/JACOBI2D 0.904, polybench/GEMVER 0.944
+LINK+FUSE warm: 3 of 6 programs more than 2 % slower than the default: polybench/JACOBI1D 0.903, polybench/JACOBI2D 0.904, polybench/GEMVER 0.944

same bits: 6 programs, 6 equal in every run of every side, 0 whose default runs differ among themselves (), 0 DIFFERENT ([])
not measured (over the time limit on the default): []
exit status not 0 on the default: []
cold runs that left objects in the cache (side: objects after each run): {'polybench/SYR2K': {'a': [], 'bc': [1, 0, 1], 'bw': [], 'cw': [], 'dw': []}}
