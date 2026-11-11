
Shares of micro-ops (`ibs_op/cnt_ctl=1/`), % of the process's samples:

| program | mode | samples | loop | decode | rows | lanes | of which generated code (decode / rows / lanes) | mem handlers | mem system | other | **removable (loop + decode + rows)** |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| hecbench/black-scholes-cuda | tier-again | 705533 | 31.9 | 23.5 | 4.0 | 9.1 | 0.0 / 0.0 / 0.0 | 0.7 | 5.0 | 25.7 | **59.4** |
| hecbench/black-scholes-cuda | tier | 590282 | 31.1 | 23.1 | 4.0 | 8.8 | 0.0 / 0.0 / 0.0 | 0.7 | 4.8 | 27.5 | **58.2** |
| polybench/LU | tier-again | 1098740 | 30.2 | 27.5 | 5.3 | 4.0 | 0.0 / 0.0 / 0.0 | 6.1 | 12.1 | 14.8 | **63.0** |
| polybench/LU | tier | 1197610 | 30.2 | 27.3 | 5.4 | 4.1 | 0.0 / 0.0 / 0.0 | 6.2 | 12.4 | 14.4 | **62.9** |
| polybench/SYRK | tier-again | 232043 | 8.6 | 8.3 | 1.9 | 2.6 | 0.0 / 0.0 / 0.0 | 19.0 | 51.1 | 8.4 | **18.8** |
| polybench/SYRK | tier | 257381 | 8.5 | 8.3 | 1.9 | 2.6 | 0.0 / 0.0 / 0.0 | 18.9 | 52.1 | 7.6 | **18.7** |
| rodinia/gaussian | tier-again | 180323 | 24.7 | 17.3 | 1.9 | 3.2 | 0.0 / 0.0 / 0.0 | 9.5 | 23.4 | 19.9 | **43.9** |
| rodinia/gaussian | tier | 169819 | 25.0 | 17.1 | 1.9 | 3.2 | 0.0 / 0.0 / 0.0 | 9.3 | 23.7 | 19.8 | **44.0** |
| rodinia/hotspot | tier-again | 47546 | 21.9 | 30.6 | 4.4 | 12.9 | 0.0 / 0.0 / 0.0 | 12.9 | 10.5 | 6.8 | **56.9** |
| rodinia/hotspot | tier | 47489 | 21.8 | 30.8 | 4.4 | 12.6 | 0.0 / 0.0 / 0.0 | 13.0 | 10.5 | 6.8 | **57.0** |

Shares of cycles (`ibs_op/cnt_ctl=0/`), % of the process's samples:

| program | mode | samples | loop | decode | rows | lanes | of which generated code (decode / rows / lanes) | mem handlers | mem system | other | **removable (loop + decode + rows)** |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| hecbench/black-scholes-cuda | tier-again | 223860 | 29.3 | 23.3 | 3.5 | 7.1 | 0.0 / 0.0 / 0.0 | 0.7 | 4.2 | 31.8 | **56.1** |
| hecbench/black-scholes-cuda | tier | 229617 | 28.6 | 23.2 | 3.5 | 7.0 | 0.0 / 0.0 / 0.0 | 0.8 | 4.7 | 32.2 | **55.3** |
| polybench/LU | tier-again | 319553 | 29.5 | 23.2 | 7.4 | 3.6 | 0.0 / 0.0 / 0.0 | 5.3 | 10.7 | 20.4 | **60.1** |
| polybench/LU | tier | 320308 | 29.7 | 23.3 | 7.5 | 3.7 | 0.0 / 0.0 / 0.0 | 5.4 | 10.7 | 19.7 | **60.5** |
| polybench/SYRK | tier-again | 68493 | 7.8 | 8.2 | 1.7 | 1.8 | 0.0 / 0.0 / 0.0 | 15.9 | 52.1 | 12.4 | **17.7** |
| polybench/SYRK | tier | 62761 | 8.2 | 8.6 | 2.0 | 2.0 | 0.0 / 0.0 / 0.0 | 16.1 | 49.6 | 13.5 | **18.8** |
| rodinia/gaussian | tier-again | 42414 | 24.1 | 16.9 | 1.9 | 3.0 | 0.0 / 0.0 / 0.0 | 9.1 | 19.0 | 25.9 | **42.9** |
| rodinia/gaussian | tier | 42448 | 23.7 | 16.6 | 1.9 | 2.9 | 0.0 / 0.0 / 0.0 | 9.1 | 19.3 | 26.4 | **42.3** |
| rodinia/hotspot | tier-again | 11617 | 24.2 | 29.5 | 5.1 | 12.6 | 0.0 / 0.0 / 0.0 | 11.4 | 8.7 | 8.6 | **58.8** |
| rodinia/hotspot | tier | 13206 | 21.9 | 30.0 | 6.8 | 11.6 | 0.0 / 0.0 / 0.0 | 12.3 | 8.5 | 9.0 | **58.7** |
