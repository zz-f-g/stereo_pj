#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace common
{
    Eigen::VectorXd nonTrivalSol(Eigen::MatrixXd &A)
    {
        using Eigen::MatrixXd;
        Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinV);
        MatrixXd V = svd.matrixV();
        return V.col(V.cols() - 1);
    }

    Eigen::VectorXd h2v(const Eigen::Vector3d &hi, const Eigen::Vector3d &hj)
    {
        using Eigen::VectorXd;
        VectorXd vij(6);
        vij << hi(0) * hj(0), hi(0) * hj(1) + hi(1) * hj(0), hi(1) * hj(1),
            hi(0) * hj(2) + hi(2) * hj(0), hi(1) * hj(2) + hi(2) * hj(1), hi(2) * hj(2);
        return vij;
    }

    Eigen::Vector3d R2rodrigues(const Eigen::Matrix3d &R)
    {
        using Eigen::Matrix3d;
        using Eigen::MatrixXd;
        using Eigen::Vector3d;
        Matrix3d temp = (R - R.transpose()) / 2.;
        Eigen::FullPivLU<MatrixXd> lu(temp);
        Vector3d axis;
        axis << (temp(2, 1) - temp(1, 2)) / 2.,
            (temp(0, 2) - temp(2, 0)) / 2.,
            (temp(1, 0) - temp(0, 1)) / 2.;
        axis.normalize();
        double theta = std::asin(temp.norm() / 2.);
        std::cout << "axis" << std::endl
                  << axis << std::endl;
        std::cout << "theta" << std::endl
                  << theta << std::endl;

        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f M;
        Eigen::Matrix3f Rk;
        Rk << 0, -axis[2], axis[1],
            axis[2], 0, -axis[0],
            -axis[1], axis[0], 0;

        M = I + (1 - cos(theta)) * Rk * Rk + sin(theta) * Rk;

        model << M(0, 0), M(0, 1), M(0, 2), 0,
            M(1, 0), M(1, 1), M(1, 2), 0,
            M(2, 0), M(2, 1), M(2, 2), 0,
            0, 0, 0, 1;
        std::cout << model << std::endl;
        exit(0);
        Vector3d res;
        res << 0, 0, 0;
        return res;
    }
}

#endif