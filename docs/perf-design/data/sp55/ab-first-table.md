| corpus | program | `VGPU_THREADS=1` s (base) | base exact 2 / 4 / 8 | this round exact 2 / 4 / 8 | `VGPU_THREADS=8` | base twice | default, new / base | trials at 8: kept / stopped / again | bits |
|---|---|---:|---|---|---:|---:|---:|---|---|
| polybench | 2DCONV | 0.72 | 0.84x / 1.00x / 1.09x | 0.98x / 1.15x / 1.21x | 1.51x | 0.993 | 0.982 | 1 / 0 / 0 | same |
| polybench | 3DCONV | 0.66 | 0.94x / 1.21x / 1.46x | 1.26x / 1.64x / 1.86x | 2.01x | 1.016 | 1.008 | 254 / 0 / 0 | same |
| polybench | 3MM | 2.06 | 0.92x / 1.25x / 1.42x | 1.24x / 1.50x / 1.66x | 1.69x | 1.004 | 1.009 | 3 / 0 / 0 | same |
| polybench | ADI | 0.23 | 0.77x / 0.93x / 0.92x | 0.82x / 1.02x / 1.02x | 1.57x | 1.004 | 1.005 | 2049 / 0 / 4 | same |
| polybench | ATAX | 1.44 | 1.08x / 1.54x / 2.28x | 1.43x / 2.27x / 3.20x | 3.95x | 1.001 | 1.015 | 2 / 0 / 0 | same |
| polybench | BICG | 0.28 | 0.85x / 1.01x / 1.18x | 1.23x / 1.53x / 1.70x | 1.86x | 1.007 | 1.011 | 2 / 0 / 0 | same |
| polybench | GEMM | 0.74 | 1.03x / 1.27x / 1.43x | 1.22x / 1.49x / 1.65x | 1.71x | 0.991 | 1.014 | 1 / 0 / 1 | same |
| polybench | GEMVER | 0.66 | 0.83x / 1.15x / 1.40x | 1.03x / 1.40x / 1.63x | 2.07x | 0.976 | 0.964 | 3 / 0 / 3 | same |
| polybench | GESUMMV | 0.39 | 0.84x / 0.98x / 1.10x | 1.14x / 1.38x / 1.54x | 1.75x | 1.014 | 1.010 | 1 / 0 / 2 | same |
| polybench | JACOBI1D | 1.08 | 0.82x / 1.03x / 0.89x | 1.05x / 1.28x / 0.83x | 1.61x | 1.004 | 0.988 | 18720 / 0 / 0 | same |
| polybench | JACOBI2D | 0.56 | 0.83x / 1.23x / 1.64x | 1.23x / 1.79x / 2.32x | 3.30x | 0.999 | 0.994 | 40 / 0 / 0 | same |
| polybench | LU | 29.48 | 1.11x / 1.70x / 2.29x | 1.24x / 1.87x / 2.49x | 3.85x | 0.985 | 0.987 | 2500 / 0 / 2 | same |
| polybench | MVT | 1.58 | 1.05x / 1.50x / 2.04x | 1.39x / 2.06x / 2.65x | 3.07x | 1.003 | 0.995 | 2 / 0 / 2 | same |
| polybench | SYR2K | 10.57 | 1.00x / 1.41x / 1.86x | 1.14x / 1.68x / 2.17x | 2.82x | 0.984 | 0.996 | 1 / 0 / 1 | same |
| polybench | SYRK | 5.50 | 1.01x / 1.23x / 1.89x | 1.15x / 1.65x / 2.22x | 2.86x | 0.978 | 0.992 | 1 / 0 / 1 | same |
| rodinia | backprop | 0.52 | 0.74x / 1.04x / 1.28x | 0.71x / 1.01x / 1.31x | 2.25x | 1.008 | 0.996 | 2 / 0 / 3 | same |
| rodinia | gaussian | 3.67 | 1.28x / 1.96x / 2.70x | 1.19x / 1.92x / 2.69x | 3.78x | 1.010 | 0.995 | 511 / 0 / 2 | same |
| rodinia | lud | 2.94 | 1.05x / 1.20x / 1.29x | 1.04x / 1.21x / 1.28x | 1.40x | 1.003 | 1.003 | 124 / 0 / 2 | same |
| rodinia | needle | 0.33 | 0.97x / 1.28x / 1.47x | 1.23x / 1.69x / 1.90x | 2.64x | 1.002 | 1.010 | 253 / 0 / 2 | same |
| rodinia | pathfinder | 0.54 | 1.08x / 1.23x / 1.28x | 1.10x / 1.23x / 1.29x | 1.34x | 1.017 | 1.001 | 5 / 0 / 1 | same |
| rodinia | srad_v2 | 0.03 | 0.95x / 0.95x / 0.93x | 0.95x / 0.92x / 0.91x | 0.92x | 0.991 | 0.994 | 0 / 1 / 0 | same |
| rodinia | particlefilter_float | 1.37 | 1.34x / 2.02x / 2.43x | 1.19x / 1.84x / 2.28x | 1.26x | 1.004 | 1.084 | 27 / 3 / 5 | not comparable (`VGPU_THREADS=1` differs from run to run) |
| rodinia | sc_gpu | 0.68 | 0.93x / 1.15x / 1.19x | 0.85x / 1.24x / 1.14x | 1.83x | 1.008 | 1.013 | 1432 / 0 / 33 | same |
| rodinia | hotspot | 0.96 | 1.39x / 2.08x / 2.73x | 1.43x / 2.20x / 2.83x | 3.29x | 1.012 | 1.005 | 1000 / 0 / 0 | same |
| rodinia | bfs | 0.01 | 0.93x / 0.96x / 0.98x | 0.98x / 1.00x / 0.92x | 1.26x | 0.889 | 0.832 | 5 / 7 / 5 | same |
| rodinia | nn | 0.01 | 0.99x / 1.02x / 0.98x | 0.99x / 1.00x / 0.99x | 1.01x | 0.947 | 0.907 | 1 / 0 / 0 | same |
| rodinia | lavaMD | 1.51 | 1.67x / 2.61x / 3.63x | 1.53x / 2.31x / 3.41x | 4.22x | 1.012 | 1.012 | 1 / 0 / 1 | same |
| hecbench | adam-cuda | 0.02 | 1.49x / 1.86x / 2.00x | 1.30x / 1.60x / 1.76x | 1.94x | 0.936 | 0.973 | 1 / 0 / 3 | same |
| hecbench | aobench-cuda | 0.57 | 1.90x / 2.86x / 4.07x | 1.81x / 2.81x / 4.03x | 4.62x | 1.002 | 1.001 | 1 / 0 / 0 | same |
| hecbench | aop-cuda | 1.78 | 1.46x / 2.19x / 2.94x | 1.52x / 2.30x / 3.08x | 3.57x | 1.023 | 1.005 | 201 / 0 / 2 | same |
| hecbench | asmooth-cuda | 0.22 | 1.04x / 1.13x / 1.19x | 1.04x / 1.13x / 1.14x | 2.87x | 1.005 | 1.025 | 2 / 1 / 1 | same |
| hecbench | asyncAllocation-cuda | 3.99 | 1.79x / 2.64x / 3.49x | 1.89x / 2.79x / 3.65x | 3.94x | 0.981 | 0.991 | 800 / 0 / 2 | same |
| hecbench | atan2-cuda | 0.02 | 1.03x / 1.23x / 1.60x | 1.13x / 1.38x / 1.66x | 1.81x | 0.961 | 0.962 | 3 / 0 / 0 | same |
| hecbench | atomicAggregate-cuda | 3.66 | 1.00x / 1.00x / 1.00x | 1.03x / 1.02x / 1.02x | 4.08x | 0.999 | 0.973 | 0 / 3 / 0 | same |
| hecbench | atomicIntrinsics-cuda | 0.18 | 1.00x / 0.99x / 0.98x | 0.99x / 1.00x / 0.98x | 3.98x | 1.001 | 0.991 | 0 / 2 / 0 | same |
| hecbench | atomicPerf-cuda | 1.06 | 1.58x / 2.12x / 2.53x | 1.59x / 2.13x / 2.84x | 3.82x | 1.003 | 1.006 | 9 / 9 / 0 | same |
| hecbench | attention-cuda | 0.02 | 0.93x / 0.87x / 0.85x | 0.91x / 0.87x / 0.87x | 1.00x | 1.014 | 0.988 | 1 / 1 / 2 | same |
| hecbench | axpby-cuda | 0.40 | 1.15x / 1.51x / 1.77x | 1.34x / 1.78x / 2.08x | 2.39x | 0.999 | 1.009 | 34 / 0 / 0 | same |
| hecbench | background-subtract-cuda | 0.00 | 1.48x / 1.58x / 1.68x | 1.62x / 1.57x / 1.53x | 1.53x | 0.672 | 0.674 | 0 / 0 / 0 | same |
| hecbench | backprop-cuda | 0.01 | 0.81x / 0.98x / 1.12x | 0.79x / 0.99x / 1.14x | 1.36x | 0.925 | 0.893 | 2 / 0 / 3 | same |
| hecbench | bh-cuda | 0.73 | 1.41x / 1.95x / 2.55x | 1.41x / 1.94x / 2.29x | 3.52x | 1.010 | 0.988 | 8 / 2 / 4 | same; `VGPU_THREADS=8` differs |
| hecbench | bilateral-cuda | 1.10 | 1.66x / 2.43x / 3.25x | 1.63x / 2.45x / 3.33x | 3.48x | 1.003 | 1.009 | 3 / 0 / 0 | same |
| hecbench | bitonic-sort-cuda | 0.26 | 1.37x / 1.97x / 2.50x | 1.24x / 1.85x / 2.42x | 3.11x | 1.012 | 1.016 | 136 / 0 / 1 | same |
| hecbench | black-scholes-cuda | 19.80 | 1.36x / 1.89x / 2.33x | 1.47x / 2.02x / 2.51x | 2.64x | 1.013 | 1.002 | 1 / 0 / 0 | same |
| hecbench | blockAccess-cuda | 0.01 | 0.86x / 1.11x / 1.27x | 0.95x / 1.21x / 1.36x | 1.59x | 0.932 | 0.960 | 4 / 0 / 0 | same |
| hecbench | perplexity-cuda | 0.37 | 1.40x / 1.93x / 1.94x | 0.73x / 1.30x / 1.30x | 2.43x | 0.991 | 1.013 | 1 / 0 / 1 | same |
| hecbench | radixsort2-cuda | 2.04 | 1.01x / 1.00x / 0.99x | 1.01x / 1.01x / 1.00x | 2.34x | 1.001 | 1.002 | 100 / 7 / 33 | same; `VGPU_THREADS=8` differs |
| hecbench | s3d-cuda | 0.38 | 0.80x / 1.08x / 1.39x | 0.94x / 1.27x / 1.56x | 2.22x | 1.007 | 1.004 | 54 / 0 / 20 | same |
| hecbench | shmembench-cuda | 4.87 | 1.80x / 2.95x / 4.14x | 1.78x / 2.94x / 4.13x | 4.27x | 1.005 | 0.994 | 1 / 0 / 0 | same |
| hecbench | silu-cuda | 0.45 | 1.34x / 1.89x / 2.51x | 1.48x / 2.15x / 2.83x | 3.26x | 1.026 | 1.020 | 50 / 0 / 0 | same |
| hecbench | sortKV-cuda | 0.18 | 0.97x / 0.97x / 0.97x | 0.98x / 0.97x / 0.96x | 1.89x | 0.996 | 1.001 | 8 / 4 / 8 | same; `VGPU_THREADS=8` differs |
| hecbench | zerocopy-cuda | 13.86 | 1.00x / 1.00x / 0.99x | 1.00x / 1.00x / 1.01x | 1.52x | 1.006 | 0.992 | 0 / 0 / 0 | same |

| geometric mean of the speed-ups | base exact 2 | 4 | 8 | this round exact 2 | 4 | 8 | `VGPU_THREADS=8` | base twice | default, new / base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all (52) | 1.10x | 1.39x | 1.63x | 1.17x | 1.51x | 1.73x | 2.29x | 0.987 | 0.985 |
| hecbench (25) | 1.23x | 1.53x | 1.78x | 1.22x | 1.54x | 1.79x | 2.56x | 0.978 | 0.977 |
| polybench (15) | 0.92x | 1.21x | 1.46x | 1.16x | 1.55x | 1.77x | 2.25x | 0.997 | 0.998 |
| rodinia (12) | 1.08x | 1.37x | 1.56x | 1.08x | 1.39x | 1.58x | 1.85x | 0.991 | 0.986 |

Summed wall seconds: `VGPU_THREADS=1` 124.5; base exact 109.1 / 83.1 / 69.3; this round exact 101.1 / 76.4 / 65.1; `VGPU_THREADS=8` 47.3.
base: 18 of 52 programs more than 3 % slower at 2 exact threads than at one: 2DCONV, 3DCONV, 3MM, ADI, BICG, GEMVER, GESUMMV, JACOBI1D, JACOBI2D, backprop, needle, srad_v2, sc_gpu, bfs, attention-cuda, backprop-cuda, blockAccess-cuda, s3d-cuda
this round: 9 of 52 programs more than 3 % slower at 2 exact threads than at one: ADI, backprop, srad_v2, sc_gpu, attention-cuda, backprop-cuda, blockAccess-cuda, perplexity-cuda, s3d-cuda
Programs of at least 0.2 s whose trials were all kept (35): 8 exact threads take 1.38x the time of 8 plain ones on the base, 1.26x this round.
Programs of at least 0.2 s (41): base twice 1.003 (geometric mean; 0.976 to 1.026); default mode, new / base 1.003 (0.964 to 1.084).
base, programs of at least 0.2 s: geometric mean 1.12x / 1.48x / 1.79x at 2 / 4 / 8 exact threads; `VGPU_THREADS=8` 2.55x
this round, programs of at least 0.2 s: geometric mean 1.21x / 1.64x / 1.93x at 2 / 4 / 8 exact threads; `VGPU_THREADS=8` 2.55x
Slower than the base by more than 3 % (programs of at least 0.2 s): JACOBI1D x8 0.93, backprop x2 0.97, backprop x4 0.97, gaussian x2 0.93, particlefilter_float x2 0.89, particlefilter_float x4 0.91, particlefilter_float x8 0.94, sc_gpu x2 0.92, sc_gpu x8 0.96, lavaMD x2 0.92, lavaMD x4 0.88, lavaMD x8 0.94, aobench-cuda x2 0.95, asmooth-cuda x8 0.96, bh-cuda x8 0.90, bitonic-sort-cuda x2 0.90, bitonic-sort-cuda x4 0.94, bitonic-sort-cuda x8 0.97, perplexity-cuda x2 0.52, perplexity-cuda x4 0.67, perplexity-cuda x8 0.67
Faster than the base by more than 3 %: 69 of 123 program-thread counts; slower: 21.
