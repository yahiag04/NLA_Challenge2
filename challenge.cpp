#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <unsupported/Eigen/SparseExtra>
#include <lis.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>

using namespace Eigen;

bool run_lis_power(LIS_MATRIX A, const double shift, double &eigenvalue,
                  LIS_INT &iterations, LIS_REAL &residual) {
    LIS_VECTOR x;
    LIS_ESOLVER solver;
    if (lis_vector_duplicate(A, &x) != LIS_SUCCESS ||
        lis_esolver_create(&solver) != LIS_SUCCESS) return false;

    std::ostringstream optionStream;
    optionStream << "-e pi -etol 1.0e-10 -shift " << std::setprecision(17) << shift << " -initx_ones true";
    const std::string options = optionStream.str();
    if (lis_esolver_set_option(const_cast<char *>(options.c_str()), solver) != LIS_SUCCESS) return false;

    LIS_SCALAR value;
    const LIS_INT solveStatus = lis_esolve(A, x, &value, solver);
    const bool ok = solveStatus == LIS_SUCCESS &&
                    lis_esolver_get_iter(solver, &iterations) == LIS_SUCCESS &&
                    lis_esolver_get_residualnorm(solver, &residual) == LIS_SUCCESS;
    if (ok) eigenvalue = static_cast<double>(value);
    lis_esolver_destroy(solver);
    lis_vector_destroy(x);
    return ok;
}

int main(int argc, char **argv) {
    // Task 1: load the grayscale image as an Eigen matrix A.
    int width, height, channels;
    unsigned char *data = stbi_load("deer.jpg", &width, &height, &channels, 1);
    if (data == nullptr) {
        printf("Error loading image: %s\n", stbi_failure_reason());
        return 1;
    }

    MatrixXd A(height, width);
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            A(i, j) = data[i * width + j];
    stbi_image_free(data);

    printf("Image loaded successfully: %d x %d\n", height, width);
    MatrixXd AtA = A.transpose() * A;
    printf("Euclidean norm of A^T A: %.12f\n", AtA.norm());

    // Task 2: solve A^T A x = lambda x and report the two largest singular values.
    SelfAdjointEigenSolver<MatrixXd> eigenSolver(AtA, EigenvaluesOnly);
    if (eigenSolver.info() != Success) {
        fprintf(stderr, "Error computing eigenvalues of A^T A\n");
        return 1;
    }
    const VectorXd eigenvalues = eigenSolver.eigenvalues();
    const double sigma1 = std::sqrt(std::max(0.0, eigenvalues[eigenvalues.size() - 1]));
    const double sigma2 = std::sqrt(std::max(0.0, eigenvalues[eigenvalues.size() - 2]));
    printf("Largest singular value: %.12f\n", sigma1);
    printf("Second largest singular value: %.12f\n", sigma2);

    // Task 3: export A^T A in Matrix Market format for the LIS eigensolver.
    SparseMatrix<double, ColMajor> AtA_sparse = AtA.sparseView(0.0, 1e-14);
    if (!saveMarket(AtA_sparse, "AtA.mtx")) {
        fprintf(stderr, "Error saving AtA.mtx\n");
        return 1;
    }
    printf("A^T A saved as AtA.mtx (%ld non-zero entries)\n", AtA_sparse.nonZeros());

    // Task 4: compare LIS power iteration with no shift and mu = lambda_2 / 2.
    int lisArgc = argc;
    char **lisArgv = argv;
    if (lis_initialize(&lisArgc, &lisArgv) != LIS_SUCCESS) {
        fprintf(stderr, "Error initializing LIS\n");
        return 1;
    }
    LIS_MATRIX lisAtA;
    if (lis_matrix_create(LIS_COMM_WORLD, &lisAtA) != LIS_SUCCESS ||
        lis_input_matrix(lisAtA, const_cast<char *>("AtA.mtx")) != LIS_SUCCESS) {
        fprintf(stderr, "Error loading AtA.mtx with LIS\n");
        lis_finalize();
        return 1;
    }

    double lisEigenvalue0, lisEigenvalueShifted;
    LIS_INT lisIterations0 = 0, lisIterationsShifted = 0;
    LIS_REAL lisResidual0 = 0.0, lisResidualShifted = 0.0;
    const double shift = 0.5 * eigenvalues[eigenvalues.size() - 2];
    if (!run_lis_power(lisAtA, 0.0, lisEigenvalue0, lisIterations0, lisResidual0) ||
        !run_lis_power(lisAtA, shift, lisEigenvalueShifted, lisIterationsShifted, lisResidualShifted)) {
        fprintf(stderr, "LIS power iteration failed\n");
        lis_matrix_destroy(lisAtA);
        lis_finalize();
        return 1;
    }
    printf("LIS largest eigenvalue (no shift): %.12f, iterations: %d, residual: %.3e\n",
           lisEigenvalue0, lisIterations0, static_cast<double>(lisResidual0));
    printf("LIS largest eigenvalue (shift mu=%.12f): %.12f, iterations: %d, residual: %.3e\n",
           shift, lisEigenvalueShifted, lisIterationsShifted, static_cast<double>(lisResidualShifted));
    lis_matrix_destroy(lisAtA);
    lis_finalize();

    // Task 5: compute the SVD and report the Euclidean norm of Sigma.
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    if (svd.info() != Success) {
        fprintf(stderr, "Error computing the SVD of A\n");
        return 1;
    }
    printf("Euclidean norm of Sigma: %.12f\n", svd.singularValues().norm());

    // Task 6: form C and D for k = 30 and k = 90, then report nonzero counts.
    for (const int k : {30, 90}) {
        if (k > svd.singularValues().size()) {
            fprintf(stderr, "k=%d exceeds the number of singular values\n", k);
            return 1;
        }
        MatrixXd C = svd.matrixU().leftCols(k);
        MatrixXd D = svd.matrixV().leftCols(k) * svd.singularValues().head(k).asDiagonal();
        printf("k=%d: C non-zero entries = %ld, D non-zero entries = %ld\n",
               k, (C.array() != 0.0).count(), (D.array() != 0.0).count());
    }
    return 0;
}
