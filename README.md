# Challenge 2 — SVD and Image Compression

Yahia Ghallale (10965385)
Enrico Jin (10890225)
Beatrice Lollio (11256228)

## Build and run

```bash
source /u/sw/etc/bash.bashrc
module load gcc-glibc
module load lis
LIS_HOME=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/lis/2.0.30
g++ -O2 -std=c++17 -DHAVE_CONFIG_H -DUSE_MPI -I"$mkEigenInc" -I"$LIS_HOME/include" challenge.cpp \
  -L"$LIS_HOME/lib" -Wl,-rpath,"$LIS_HOME/lib" -llis -lmpi -o challenge
./challenge
```

`deer.jpg` needs to be in the same directory. Running it writes `AtA.mtx`, the compressed deer images, the checkerboards, and their rank-6/rank-12 reconstructions.

## Results

**Deer image** (656 × 656):

| Task | Result |
| --- | --- |
| 1 | `‖A^T A‖` = 1,210,497,678.444907188416 |
| 2 | σ1 = 34,645.143595853711, σ2 = 10,623.769487838059 |
| 3 | `AtA.mtx` exported, 430,336 nonzero entries |
| 4 | LIS eigenvalue = 1,200,285,974.777324438095 — no shift: 10 iter, residual 8.349e-11; shift `mu = λ2/2` = 56,432,239.065359473228: 9 iter, residual 7.195e-12 |
| 5 | `‖Σ‖` = 41,415.341203476681 |
| 6 | k=30: C, D with 19,680 nonzero entries each; k=90: 59,040 each |

The shifted run gives the same eigenvalue (1,200,285,974.777323246002) one iteration sooner.

**Checkerboard** (400 × 400, 40 × 40 squares):

- Frobenius norm: 72124.891681
- Noisy image singular values: σ1 = 51011.909045898668, σ2 = 51002.421720600963
- k=6 and k=12: C, D are 400 × 6 and 400 × 12

| | RMSE vs original | RMSE vs noisy |
| --- | ---: | ---: |
| Noisy checkerboard | 34.586243358632 | 0 |
| Rank 6 | 7.513007859763 | 33.760514094702 |
| Rank 12 | 10.934733714287 | 32.812322131300 |

The clean board is rank 2, so rank 6 already overshoots it and rank 12 captures even more noise — hence rank 6 is closer to the original than rank 12.

![Original checkerboard](checkerboard.png)
![Noisy checkerboard](noisy_checkerboard.png)
![Rank-6 reconstruction](compressed_checkerboard_k6.png)
![Rank-12 reconstruction](compressed_checkerboard_k12.png)

## Tests

`test_challenge.cpp` includes `challenge.cpp` and asserts on `build_matrices`, `create_checkerboard`, and `write_image`. Build the same way as above with `test_challenge.cpp` in place of `challenge.cpp`, without `-DNDEBUG`.
