#include <iostream>
#include <Eigen/Dense>

int main()
{
    Eigen::MatrixXd A(3, 3);
    A << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    std::cout << "U:\n"
              << U << std::endl;
    std::cout << "V:\n"
              << V << std::endl;
    std::cout << svd.singularValues() << std::endl;

    return 0;
}
