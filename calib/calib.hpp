#ifndef CALIB_HPP
#define CALIB_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <stdlib.h>
#include <opencv2/core/eigen.hpp>
#include "common.hpp"

namespace calib
{
    struct calibCallback : public cv::LMSolver::Callback
    {
        std::vector<std::vector<cv::Point3f>> op; // objectPoints
        std::vector<std::vector<cv::Point2f>> ip; // imagePoints
        cv::Size is;                              // imageSize
        uint32_t nImgs;

        calibCallback() = default;

        virtual bool compute(cv::InputArray f_param, cv::OutputArray f_error, cv::OutputArray f_jacobian) const override
        {
            using Eigen::Vector2d;
            using Eigen::MatrixXd;
            cv::Mat param = f_param.getMat();
            MatrixXd p;
            cv::cv2eigen(param, p);
            std::cout << "p eigen:" << std::endl;
            std::cout << p << std::endl;
            exit(0);

            if (f_error.empty())
                f_error.create(1, 1, CV_64F); // dim(error) = 1
            cv::Mat error = f_error.getMat();
            double error0 = 0.;

            for (uint32_t idxImg; idxImg < nImgs; idxImg++)
            {
                std::vector<cv::Point3f> objPts = op.at(idxImg);
                std::vector<cv::Point2f> imgPts = ip.at(idxImg);
                uint32_t nPts = objPts.size();
                for (uint32_t idxPt; idxPt < nPts; idxPt++)
                {
                    Vector2d imgPtVec;
                    imgPtVec(0) = objPts.at(idxPt).x;
                    imgPtVec(1) = objPts.at(idxPt).y;
                    // error0 += (imgPtVec - this->calc(A, R, T, objPtsMat)).norm();
                }
            }

            error.at<double>(0, 0) = error0;

            /*
            if (!f_jacobian.needed())
                return true;
            else if (f_jacobian.empty())
                f_jacobian.create(1, 2, CV_64F);
            cv::Mat jacobian = f_jacobian.getMat();

            double e = 1e-10;                                                  // estimate derivatives in epsilon environment
            jacobian.at<double>(0, 0) = (calc({x[0] + e, x[1]}) - error0) / e; // d/dx0 (error)
            jacobian.at<double>(0, 1) = (calc({x[0], x[1] + e}) - error0) / e; // d/dx1 (error)
            */
            return true;
        }

        Eigen::MatrixXd calc(Eigen::Matrix3d &A, Eigen::Matrix3d &R, Eigen::Vector3d &T, Eigen::MatrixXd &objectPoints) const
        {
            using Eigen::MatrixXd;
            using Eigen::Matrix3d;
            using Eigen::Vector3d;
            MatrixXd imagePoints;
            return imagePoints;
        }
    };
    cv::Mat initIntrinsic(const std::vector<std::vector<cv::Point3f>> &objectPoints,
                          const std::vector<std::vector<cv::Point2f>> &imagePoints,
                          cv::Size &imageSize)
    {

        // CLOSED-FORM SOLUTION
        // 1. get homogeneous matrix from objectPoints to imagePoints
        using Eigen::Matrix3d;
        using Eigen::MatrixXd;
        using Eigen::Vector3d;
        using Eigen::VectorXd;
        uint32_t nImgs = objectPoints.size();
        MatrixXd H(3, 3 * nImgs); // homogeneous matrix for each image
        MatrixXd V(2 * nImgs, 6); // matrix to compute A^{-T} A
        for (uint32_t idxImg = 0; idxImg < nImgs; idxImg++)
        {
            std::vector<cv::Point3f> objPts = objectPoints.at(idxImg);
            std::vector<cv::Point2f> imgPts = imagePoints.at(idxImg);
            uint32_t nPts = objPts.size();
            // 1.1 get matrix from a pair of points
            // 1.1.1 homogeneous coordinates of objPts
            MatrixXd matFromPts(2 * nPts, 9);
            for (uint32_t idxPt = 0; idxPt < nPts; idxPt++)
            {
                // 1.1.2 build matrix [M',0,-u*M';0,M',-v*M']
                // todo: normalization
                MatrixXd matFromPt(2, 9);
                matFromPt << objPts.at(idxPt).x, objPts.at(idxPt).y, 1.,
                    0., 0., 0,
                    -imgPts.at(idxPt).x / double(imageSize.width) * objPts.at(idxPt).x,
                    -imgPts.at(idxPt).x / double(imageSize.width) * objPts.at(idxPt).y,
                    -imgPts.at(idxPt).x / double(imageSize.width), // normalize
                    0., 0., 0,
                    objPts.at(idxPt).x, objPts.at(idxPt).y, 1.,
                    -imgPts.at(idxPt).y / double(imageSize.height) * objPts.at(idxPt).x,
                    -imgPts.at(idxPt).y / double(imageSize.height) * objPts.at(idxPt).y,
                    -imgPts.at(idxPt).y / double(imageSize.height); // normalize
                matFromPts.row(2 * idxPt) = matFromPt.row(0);
                matFromPts.row(2 * idxPt + 1) = matFromPt.row(1);
            }
            /*
            std::cout << "mat from points in image" << std::endl
                      << matFromPts << std::endl;
            exit(0);
            */
            // 1.2 compute h row expanded as the last column of svd V
            VectorXd hRowExpanded = common::nonTrivalSol(matFromPts);
            /*
            std::cout << "h row expanded" << std::endl
                      << hRowExpanded.rows() << "x" << hRowExpanded.cols() << std::endl
                      << hRowExpanded << std::endl;
            */
            Eigen::Vector3d h1, h2, h3;
            h1(0) = hRowExpanded(0);
            h1(1) = hRowExpanded(3);
            h1(2) = hRowExpanded(6);
            h2(0) = hRowExpanded(1);
            h2(1) = hRowExpanded(4);
            h2(2) = hRowExpanded(7);
            h3(0) = hRowExpanded(2);
            h3(1) = hRowExpanded(5);
            h3(2) = hRowExpanded(8);
            H.col(idxImg * 3) = h1;
            H.col(idxImg * 3 + 1) = h2;
            H.col(idxImg * 3 + 2) = h3;
            /*
            std::cout << "H:" << std::endl
                      << H << std::endl;
            std::cout << "h1" << std::endl
                      << h1 << std::endl;
            std::cout << "h2" << std::endl
                      << h2 << std::endl;
            std::cout << "h3" << std::endl
                      << h3 << std::endl;
            exit(0);
            */

            // 2. from h1, h2 get v11, v12, v22
            VectorXd v11(6);
            v11 = common::h2v(h1, h1);
            VectorXd v22(6);
            v22 = common::h2v(h2, h2);
            VectorXd v12(6);
            v12 = common::h2v(h1, h2);

            V.row(2 * idxImg) = v12;
            V.row(2 * idxImg + 1) = v11 - v22;
        }

        /*
                std::cout << "H:" << std::endl
                          << H << std::endl;
                exit(0);*/
        /*
        std::cout << "V:" << std::endl
                  << V << std::endl;
                  */
        // 3. from \boldsymbol{V} \boldsymbol{b} = \boldsymbol{0} get b
        VectorXd b = common::nonTrivalSol(V);

        /*
        std::cout << "b:" << std::endl
                  << b << std::endl;*/

        // 4. from b get closed-form intrinsic
        Matrix3d A(3, 3);
        A(1, 0) = 0.;
        A(2, 0) = 0.;
        A(2, 1) = 0.;
        A(2, 2) = 1.;

        A(1, 2) = (b(1) * b(3) - b(0) * b(4)) / (b(0) * b(2) - b(1) * b(1)); // v0
        double lamda = b(5) - (b(3) * b(3) + A(1, 2) * (b(1) * b(3) - b(0) * b(4))) / b(0);
        A(0, 0) = std::sqrt(lamda / b(0));                                        // alpha
        A(1, 1) = std::sqrt(lamda * b(0) / (b(0) * b(2) - b(1) * b(1)));          // beta
        A(0, 1) = -b(1) * A(0, 0) * A(0, 0) * A(1, 1) / lamda;                    // gamma
        A(0, 2) = A(0, 1) * A(1, 2) / A(0, 0) - b(3) * A(0, 0) * A(0, 0) / lamda; // u0

        A(0, 0) *= double(imageSize.width);
        A(0, 1) *= double(imageSize.width);
        A(0, 2) *= double(imageSize.width);
        A(1, 1) *= double(imageSize.height);
        A(1, 2) *= double(imageSize.height);

        MatrixXd r1r2t(3, 3 * nImgs);
        MatrixXd rotate(3, nImgs); // rodrigues rotate: theta, t
        MatrixXd T(3, nImgs);
        Matrix3d R;
        for (uint32_t idxImg = 0; idxImg < nImgs; idxImg++)
        {
            R.col(0) = A.inverse() * H.col(3 * idxImg);
            R.col(1) = A.inverse() * H.col(3 * idxImg + 1);
            R.col(2) = R.col(0).cross(R.col(1));
            T.col(idxImg) = A.inverse() * H.col(3 * idxImg + 2);
            /*std::cout << "R" << std::endl
                      << R << std::endl;*/
            R.col(0).normalize();
            R.col(1).normalize();
            R.col(2) = R.col(0).cross(R.col(1));
            std::cout << "Img " << idxImg << ": r1' * r2 = " << R.col(0).dot(R.col(1)) << std::endl;
            /*std::cout << "R" << std::endl
                      << R << std::endl;*/
            /*
            common::R2rodrigues(R);*/
        }
        exit(0);
        std::cout << "!r1r2t:" << std::endl
                  << r1r2t << std::endl;
        std::cout << "intrinsic" << std::endl
                  << A << std::endl;
        /*
        Matrix3d R = r1r2t;
        MatrixXd T(3, nImgs);
        for (uint32_t idxImg; idxImg < nImgs; idxImg++)
        {
            Vector3d r1 = r1r2t.col(3 * idxImg);
            Vector3d r2 = r1r2t.col(3 * idxImg + 1);
            T.col(idxImg) = r1r2t.col(3 * idxImg + 2); // t
            R.col(3 * idxImg + 2) = r1.cross(r2); // r3
        }

        std::cout << "Intrinsic before refine:" << std::endl
                  << A << std::endl;*/

        // MAXIMUM-LIKELIHOOD ESTIMATION
        // 1. initiate callback function and LM solver
        /*
        cv::Ptr<calibCallback> callback = cv::makePtr<calibCallback>();
        callback->op = objectPoints;
        callback->ip = imagePoints;
        callback->is = imageSize;
        callback->nImgs = nImgs;

        cv::Ptr<cv::LMSolver> solver = cv::LMSolver::create(callback, 100000, 1e-37);

        // 2. build parameters matrix
        */
        cv::Mat intrinsic(3, 3, CV_64F);
        cv::eigen2cv(A, intrinsic);

        /*
        cv::Mat rotate(3, nImgs, CV_64F);
        cv::eigen2cv(R, rotate);

        cv::Mat translate(3, nImgs, CV_64F);
        cv::eigen2cv(T, translate);

        cv::Mat param(3, 4 * nImgs + 3, CV_64F);
        cv::Mat roiA = param(cv::Rect(0, 0, 3, 3));
        cv::Mat roiR = param(cv::Rect(3, 0, 3 * nImgs, 3));
        cv::Mat roiT = param(cv::Rect(3 * nImgs + 3, 0, nImgs, 3));
        intrinsic.copyTo(roiA);
        rotate.copyTo(roiR);
        translate.copyTo(roiT);
        std::cout << "parameters:" << std::endl
                  << param << std::endl;
        exit(0);
        cv::Mat parameters = (Mat_<double>(2, 1) << 5, 100);
        // 3. run LM solver
        solver->run(param);
        std::cout << param << std::endl;

        */
        return intrinsic;
    }
}

#endif