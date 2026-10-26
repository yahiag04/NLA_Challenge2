#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
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
#include <vector>

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

void create_checkerboard(MatrixXd& board);
unsigned char* write_image(const MatrixXd& image);
bool build_matrices(int k, MatrixXd& C, MatrixXd& D, JacobiSVD<MatrixXd>& svd);




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
    MatrixXd C1, D1, C2, D2; 

    if (!build_matrices(30, C1, D1, svd) ||
        !build_matrices(90, C2, D2, svd)) return 1;

    //Task 7
    MatrixXd image_1 = C1 * D1.transpose();
    data = write_image(image_1);

    if(stbi_write_png("compressed_image_1.png", image_1.cols(), image_1.rows(), 1, data, image_1.cols()) == 0) {
        printf("Error writing image");
        delete[] data;
        return 1;
    }else {
        printf("Compressed image saved successfully as compressed_image_1.png\n");
    }
    delete[] data;



    MatrixXd image_2 = C2 * D2.transpose();
    data = write_image(image_2);

    if(stbi_write_png("compressed_image_2.png", image_2.cols(), image_2.rows(), 1, data, image_2.cols()) == 0) {
        printf("Error writing image");
        delete[] data;
        return 1;
    }else {
        printf("Compressed image saved successfully as compressed_image_2.png\n");
    }
    delete[] data;

    

    //Task 8
    MatrixXd checkerboard(400, 400);
    create_checkerboard(checkerboard);

    printf("The norm of the board matrix is: %f\n", checkerboard.norm());

    data = write_image(checkerboard);
    const int boardSaved = stbi_write_png("checkerboard.png", 400, 400, 1, data, 400);
    delete[] data;
    if (!boardSaved) {
        fprintf(stderr, "Error writing checkerboard.png\n");
        return 1;
    }

    //Task 9
    MatrixXd noisy_checkerboard = checkerboard;
    
    for(int i = 0; i < noisy_checkerboard.rows(); i++) {
        for(int j = 0; j < noisy_checkerboard.cols(); j++) {
            double noise = ((double)rand() / RAND_MAX) * 120 - 60;
            noisy_checkerboard(i, j) += noise;
            if(noisy_checkerboard(i, j) < 0) noisy_checkerboard(i, j) = 0; 
            if(noisy_checkerboard(i, j) > 255) noisy_checkerboard(i, j) = 255;
        }
    }

    data = write_image(noisy_checkerboard);

    if(stbi_write_png("noisy_checkerboard.png", noisy_checkerboard.cols(), noisy_checkerboard.rows(), 1, data, noisy_checkerboard.cols()) == 0) {
        printf("Error writing image");
        delete[] data;
        return 1;
    }else {
        printf("Noisy chckerboard saved successfully as noisy_checkerboard.png\n");
    }
    delete[] data;



    // Task 10: SVD of the noisy image, after clipping pixels to [0, 255].
    JacobiSVD<MatrixXd> noisySvd(noisy_checkerboard, ComputeThinU | ComputeThinV);
    if (noisySvd.info() != Success) {
        fprintf(stderr, "Error computing the SVD of the noisy checkerboard\n");
        return 1;
    }
    printf("Noisy checkerboard largest singular values: %.12f, %.12f\n",
           noisySvd.singularValues()(0), noisySvd.singularValues()(1));

    std::vector<MatrixXd> reconstructions;
    for (const int k : {6, 12}) {
        // Task 11: both factors have size 400 x k.
        MatrixXd C, D;
        if (!build_matrices(k, C, D, noisySvd)) return 1;
        printf("Checkerboard k=%d: C size = %ld x %ld, D size = %ld x %ld\n",
               k, C.rows(), C.cols(), D.rows(), D.cols());

        // Task 12: reconstruct and export each rank-k approximation.
        const MatrixXd compressed = C * D.transpose();
        const std::string filename = "compressed_checkerboard_k" + std::to_string(k) + ".png";
        data = write_image(compressed);
        const int saved = stbi_write_png(filename.c_str(), compressed.cols(),
                                        compressed.rows(), 1, data, compressed.cols());
        delete[] data;
        if (!saved) {
            fprintf(stderr, "Error writing %s\n", filename.c_str());
            return 1;
        }
        reconstructions.push_back(compressed);
        printf("Compressed checkerboard saved as %s\n", filename.c_str());
    }

    // Task 13: compare floating-point reconstructions before PNG quantization.
    printf("Noisy checkerboard RMSE versus original: %.12f\n",
           (noisy_checkerboard - checkerboard).norm() / std::sqrt(checkerboard.size()));
    std::size_t i = 0;
    for (const int k : {6, 12}) {
        const MatrixXd& compressed = reconstructions[i++];
        printf("Checkerboard k=%d: RMSE versus original = %.12f, versus noisy = %.12f\n",
               k, (compressed - checkerboard).norm() / std::sqrt(checkerboard.size()),
               (compressed - noisy_checkerboard).norm() / std::sqrt(checkerboard.size()));
    }
    return 0;
}


void create_checkerboard(MatrixXd& board){
    int square = 40;

    for(int i = 0; i < board.rows(); i++){
        for(int j = 0; j < board.cols(); j++){
            board(i, j) = (((i / square) + (j / square)) % 2 == 0) ? 0 : 255;
        }
    }
}

unsigned char* write_image(const MatrixXd& image){
    auto height = image.rows();
    auto width = image.cols();

    unsigned char* data = new unsigned char[height * width];

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            data[i * width + j] = static_cast<unsigned char>(std::clamp(image(i, j), 0.0, 255.0));
        }
    }

    return data;
}

bool build_matrices(int k, MatrixXd& C, MatrixXd& D, JacobiSVD<MatrixXd>& svd){

    if (k < 1 || k > svd.singularValues().size()) {
        fprintf(stderr, "Invalid k=%d: expected 1 <= k <= %ld\n", k, svd.singularValues().size());
        return false;
    }
    C = svd.matrixU().leftCols(k);
    D = svd.matrixV().leftCols(k) * svd.singularValues().head(k).asDiagonal();
    printf("k=%d: C non-zero entries = %ld, D non-zero entries = %ld\n",
            k, (C.array() != 0.0).count(), (D.array() != 0.0).count());
    return true;
}
