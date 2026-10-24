# Challenge 2 — SVD and Image Compression

**Authors**

- Yahia Ghallale (10965385)
- Enrico Jin (10890225)
- Beatrice Lollio (11256228)

This project completes tasks 1–6 of the Numerical Linear Algebra Challenge 2. The implementation follows the Challenge 1 pattern: numbered task blocks in `challenge.cpp`, Eigen matrices and vectors, `stb_image` image input, and printed results. It uses the grayscale `deer.jpg` from Challenge 1.

## Input and generated files

Place `deer.jpg` in the project directory. Run `challenge` from that directory. It reads the image, writes `AtA.mtx`, runs the LIS eigenvalue calculation, and prints the results for tasks 1–6.

## Tasks completed

1. Load the image into Eigen matrix `A` and report the Euclidean norm of `A^T A`.
2. Solve the symmetric eigenvalue problem for `A^T A`; report the two largest singular values as the square roots of its two largest eigenvalues.
3. Export `A^T A` as `AtA.mtx` in Matrix Market coordinate format for LIS.
4. Run LIS's power eigensolver both without a shift and with `mu = lambda2 / 2`, and report the eigenvalue and iteration count for both runs. The shifted run also prints its residual.
5. Compute the thin SVD of `A` with Eigen and report `||Sigma||_2` as the Euclidean norm of its singular-value vector.
6. Form `C = U_k` and `D = V_k Sigma_k` for `k = 30` and `k = 90`, and report their nonzero entry counts.

## Results

The grayscale deer image is **656 × 656** pixels. The following values were reported by the completed run:

| Task | Result |
| --- | --- |
| 1 | `A^T A` Euclidean norm: `1,210,497,678.444907188416` |
| 2 | Largest singular value: `34,645.143595853711`; second largest: `10,623.769487838059` |
| 3 | `A^T A` exported to `AtA.mtx`; `430,336` nonzero entries |
| 4 | LIS largest eigenvalue: `1,200,285,974.777324438095`; no shift: 10 iterations, residual `8.349e-11`; shift `mu = 56,432,239.065359473228`: 9 iterations, residual `7.195e-12` |
| 5 | Euclidean norm of `Sigma`: `41,415.341203476681` |
| 6 | `k=30`: `C` and `D` each have `19,680` nonzero entries; `k=90`: each has `59,040` |

The shifted LIS eigenvalue was `1,200,285,974.777323246002`, agreeing with the unshifted value to the displayed precision. The shift reduced the iteration count by one.

## Build and run

Use the course container, as for Challenge 1:

```bash
source /u/sw/etc/bash.bashrc
module load gcc-glibc
module load lis
LIS_HOME=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/lis/2.0.30
g++ -O2 -std=c++17 -DHAVE_CONFIG_H -DUSE_MPI -I"$mkEigenInc" -I"$LIS_HOME/include" challenge.cpp \
  -L"$LIS_HOME/lib" -Wl,-rpath,"$LIS_HOME/lib" -llis -lmpi -o challenge
./challenge
```

Task 4 runs through LIS's C API from `challenge.cpp`; it does not require `etest5` or a manual copy of `AtA.mtx`. The program uses `mu = lambda2 / 2` and prints the baseline and shifted iteration counts for comparison. LIS reports the original largest eigenvalue alongside its shifted iteration run.

## Notes

`A^T A` is 656 × 656 for the deer image (at most 430,336 entries), so its Matrix Market export is practical. The Eigen eigenvalue and SVD calculations are dense and may take longer than the image operations.
