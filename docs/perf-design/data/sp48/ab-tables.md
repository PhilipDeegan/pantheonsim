| program | tier s (spread) | JIT s (spread) | JIT again s (spread) | fused s (spread) | **JIT / fused** | control JIT / JIT again | in regions M | generated M | region calls M | outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| rodinia/hotspot | 0.998 (3%) | 0.885 (2%) | 0.889 (6%) | 0.861 (2%) | **1.028** | 0.996 | 47.2 | 38.3 | 3.35 | same |
| rodinia/gaussian | 3.587 (1%) | 3.358 (3%) | 3.362 (1%) | 3.314 (4%) | **1.014** | 0.999 | 146.2 | 137.6 | 15.45 | same |
| polybench/SYRK | 5.395 (1%) | 4.892 (3%) | 4.886 (2%) | 4.683 (1%) | **1.045** | 1.001 | 194.2 | 91.3 | 0.10 | same |

geometric mean JIT / fused: **1.0287** over 3 programs (1.014 to 1.045); control: 0.9987 (0.996 to 1.001)

| program | JIT s (spread) | fused s (spread) | **JIT / fused** | in regions M | generated M | region calls M | outputs |
|---|---:|---:|---:|---:|---:|---:|---|
| polybench/LU | 24.367 (24%) | 23.907 (2%) | **1.019** | 1594.1 | 1456.4 | 136.62 | same |
| hecbench/black-scholes-cuda | 20.106 (1%) | 18.104 (1%) | **1.111** | 910.8 | 798.4 | 115.61 | same |

geometric mean JIT / fused: **1.0639** over 2 programs (1.019 to 1.111)
