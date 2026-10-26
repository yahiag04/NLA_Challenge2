#define main challenge_main
#include "challenge.cpp"
#undef main
#include <cassert>

int main() {
    MatrixXd input = MatrixXd::Identity(2, 2);
    JacobiSVD<MatrixXd> svd(input, ComputeThinU | ComputeThinV);
    MatrixXd C = MatrixXd::Constant(1, 1, 7.0), D = C;
    // Invalid ranks must leave the outputs untouched, without accessing Eigen blocks.
    build_matrices(3, C, D, svd);
    assert(C.rows() == 1 && C(0, 0) == 7.0 && D(0, 0) == 7.0);
    build_matrices(-1, C, D, svd);
    assert(C.rows() == 1 && C(0, 0) == 7.0 && D(0, 0) == 7.0);
    build_matrices(2, C, D, svd);
    assert((C * D.transpose() - input).norm() < 1e-12);

    MatrixXd board(400, 400);
    create_checkerboard(board);
    assert((board.array() == 255.0).count() == 80000);
    assert(std::abs(board.norm() - 255.0 * std::sqrt(80000.0)) < 1e-8);
    assert(board(0, 0) == 0 && board(0, 40) == 255 && board(40, 40) == 0);
    MatrixXd pixels(2, 2);
    pixels << -10, 100, 200, 300;
    const MatrixXd original = pixels;
    unsigned char* bytes = write_image(pixels);
    assert(bytes[0] == 0 && bytes[1] == 100 && bytes[2] == 200 && bytes[3] == 255);
    delete[] bytes;
    assert(pixels == original);
    puts("All regression checks passed");
}
