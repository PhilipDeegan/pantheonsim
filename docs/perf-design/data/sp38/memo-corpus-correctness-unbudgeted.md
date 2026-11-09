| program | launches | hits cold | hits warm | verified | verify mismatches | off = cold = warm = verify (stdout, files, exit code, digests, counters, final memory) |
|---|---:|---:|---:|---:|---:|---|
| hecbench/adam-cuda | 1 | 0 | 1 | 1 | 0 | identical |
| hecbench/aobench-cuda | 1 | 0 | 1 | 1 | 0 | identical |
| hecbench/aop-cuda | 301 | 0 | 301 | 301 | 0 | identical |
| hecbench/asmooth-cuda | 3 | 0 | 3 | 3 | 0 | identical |
| hecbench/asyncAllocation-cuda | 800 | 0 | 800 | 800 | 0 | identical |
| hecbench/atomicAggregate-cuda | 6 | 0 | 6 | 6 | 0 | identical |
| hecbench/atomicIntrinsics-cuda | 4 | 0 | 4 | 4 | 0 | identical |
| hecbench/atomicPerf-cuda | 18 | 0 | 18 | 18 | 0 | identical |
| hecbench/axpby-cuda | 35 | 0 | 0 | 0 | 0 | identical |
| hecbench/backprop-cuda | 2 | 0 | 1 | 1 | 0 | identical (nondeterministic without the cache: digest) |
| hecbench/bgmv-cuda | 12 | 0 | 12 | 12 | 0 | identical |
| hecbench/bilateral-cuda | 3 | 0 | 3 | 3 | 0 | identical |
| hecbench/bitonic-sort-cuda | 136 | 0 | 136 | 136 | 0 | identical |
| hecbench/radixsort2-cuda | 600 | 0 | 600 | 600 | 0 | identical |
| hecbench/s3d-cuda | 54 | 0 | 54 | 54 | 0 | identical |
| hecbench/silu-cuda | 50 | 0 | 50 | 50 | 0 | identical |
| hecbench/sortKV-cuda | 48 | 0 | 48 | 48 | 0 | identical |
| polybench/3MM | 3 | 0 | 3 | 3 | 0 | identical |
| polybench/ADI | 2049 | 0 | 2049 | 2049 | 0 | identical |
| polybench/JACOBI1D | 20000 | 0 | 20000 | 20000 | 0 | identical |
| polybench/JACOBI2D | 40 | 0 | 40 | 40 | 0 | identical |
| rodinia/backprop | 2 | 0 | 1 | 1 | 0 | identical (nondeterministic without the cache: digest) |
| rodinia/bfs | 26 | 0 | 26 | 26 | 0 | identical |
| rodinia/gaussian | 1022 | 0 | 1022 | 1022 | 0 | identical |
| rodinia/hotspot | 1000 | 0 | 1000 | 1000 | 0 | identical |
| rodinia/lud | 190 | 0 | 190 | 190 | 0 | identical |
| rodinia/needle | 255 | 0 | 255 | 255 | 0 | identical |
| rodinia/nn | 1 | 0 | 1 | 1 | 0 | identical |
| rodinia/particlefilter_float | 36 | 0 | 0 | 0 | 0 | identical (nondeterministic without the cache: out, digest, counters) |
| rodinia/sc_gpu | 1432 | 0 | 0 | 0 | 0 | identical (nondeterministic without the cache: digest) |
| samples/LargeKernelParameter | 2002 | 999 | 2002 | 2002 | 0 | identical |
| samples/MC_EstimatePiInlineP | 2 | 0 | 2 | 2 | 0 | identical |
| samples/MC_SingleAsianOptionP | 3 | 0 | 3 | 3 | 0 | identical |
| samples/binaryPartitionCG | 1 | 0 | 1 | 1 | 0 | identical |
| samples/cdpQuadtree | 6 | 0 | 5 | 4 | 0 | identical |
| samples/clock | 1 | 0 | 1 | 1 | 0 | identical |
| samples/convolutionSeparable | 34 | 30 | 34 | 34 | 0 | identical |
| samples/cudaGraphsPerfScaling | 213 | 193 | 195 | 195 | 0 | identical (nondeterministic without the cache: out) |
| samples/cudaOpenMP | 1 | 0 | 1 | 1 | 0 | identical |
| samples/dct8x8 | 118 | 99 | 99 | 99 | 0 | identical |
| samples/histogram | 68 | 60 | 68 | 68 | 0 | identical |
| samples/inlinePTX | 1 | 0 | 1 | 1 | 0 | identical |
| samples/mergeSort | 49 | 0 | 49 | 49 | 0 | identical |
| samples/newdelete | 6 | 0 | 0 | 0 | 0 | identical |
| samples/ptxjit | 1 | 0 | 1 | 1 | 0 | identical |
| samples/reduction | 201 | 0 | 201 | 201 | 0 | identical |
| samples/shfl_scan | 5 | 0 | 5 | 5 | 0 | identical |
| samples/simpleAtomicIntrinsics | 1 | 0 | 1 | 1 | 0 | identical |
| samples/simpleCallback | 8 | 0 | 2 | 2 | 0 | **DIFFERS: cold.digest, warm.digest, verify.digest** |
| samples/simpleCudaGraphs | 24 | 0 | 24 | 24 | 0 | identical |
| samples/simpleHyperQ | 65 | 0 | 65 | 65 | 0 | identical |
| samples/simpleTemplates | 2 | 0 | 2 | 2 | 0 | identical |
| samples/simpleVoteIntrinsics | 3 | 0 | 3 | 3 | 0 | identical |
| samples/sortingNetworks | 290 | 0 | 290 | 290 | 0 | identical |
| samples/streamOrderedAllocation | 21 | 0 | 21 | 21 | 0 | identical |
| samples/template | 1 | 0 | 1 | 1 | 0 | identical |
| samples/threadFenceReduction | 101 | 0 | 101 | 101 | 0 | identical |
| samples/transpose | 808 | 792 | 808 | 808 | 0 | identical |

58 programs compared, 1 differ; warm hits 30611 of 32166 launches (95.2 %); 30610 hits verified by a real run, 0 mismatches.
