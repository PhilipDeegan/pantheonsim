
Shares of micro-ops (`ibs_op/cnt_ctl=1/`), % of the process's samples:

| program | mode | samples | loop | decode | rows | lanes | of which generated code (decode / rows / lanes) | mem handlers | mem system | other | **removable (loop + decode + rows)** |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| hecbench/black-scholes-cuda | jit | 638808 | 39.5 | 12.2 | 4.1 | 10.1 | 7.8 / 3.4 / 8.1 | 0.8 | 5.9 | 27.3 | **55.8** |
| polybench/LU | jit | 890339 | 38.7 | 10.7 | 4.2 | 3.9 | 10.7 / 4.2 / 3.9 | 7.8 | 15.8 | 19.0 | **53.6** |
| polybench/SYRK | jit | 223920 | 0.1 | 4.3 | 1.9 | 2.4 | 4.3 / 1.9 / 2.4 | 21.0 | 61.5 | 8.8 | **6.3** |
| rodinia/gaussian | jit | 158033 | 27.2 | 7.6 | 1.4 | 4.2 | 7.6 / 1.4 / 4.2 | 10.6 | 24.9 | 24.1 | **36.2** |
| rodinia/hotspot | jit | 39102 | 23.7 | 19.8 | 4.3 | 14.5 | 5.4 / 2.4 / 2.5 | 16.0 | 12.8 | 8.9 | **47.8** |

Shares of cycles (`ibs_op/cnt_ctl=0/`), % of the process's samples:

| program | mode | samples | loop | decode | rows | lanes | of which generated code (decode / rows / lanes) | mem handlers | mem system | other | **removable (loop + decode + rows)** |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| hecbench/black-scholes-cuda | jit | 225069 | 31.3 | 14.2 | 5.3 | 10.5 | 9.9 / 4.6 / 9.1 | 0.7 | 4.0 | 34.0 | **50.8** |
| polybench/LU | jit | 256073 | 36.7 | 11.3 | 4.7 | 3.0 | 11.3 / 4.7 / 3.0 | 7.0 | 12.8 | 24.5 | **52.7** |
| polybench/SYRK | jit | 57068 | 0.3 | 4.7 | 3.2 | 2.9 | 4.7 / 3.2 / 2.9 | 18.9 | 55.1 | 14.8 | **8.2** |
| rodinia/gaussian | jit | 38908 | 26.5 | 7.5 | 1.7 | 4.5 | 7.5 / 1.7 / 4.5 | 10.7 | 20.6 | 28.5 | **35.7** |
| rodinia/hotspot | jit | 10413 | 25.7 | 19.9 | 5.8 | 14.7 | 7.1 / 4.1 / 3.7 | 13.3 | 10.3 | 10.4 | **51.3** |
