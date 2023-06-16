#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <eigen3/Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>



#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <sstream>

using namespace cv;
using namespace std;
using namespace Eigen;


void showPointCloud(
    const vector<Vector3d, Eigen::aligned_allocator<Vector3d>> &pointcloud);
void Disp2PointCloud(
    const cv::Mat disparity, const cv::Mat left, const cv::Mat right, 
    double fx, double fy, 
    double u0, double v0, double baseline);



static void print_help(char** argv)
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: %s <left_image> <right_image> [--algorithm=bm|sgbm|hh|hh4|sgbm3way] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
           "[--no-display] [--color] [-o=<disparity_image>] [-p=<point_cloud_file>]\n", argv[0]);
}



int main(int argc, char** argv)
{
    std::string img1_filename = "";
    std::string img2_filename = "";
    std::string intrinsic_filename = "";
    std::string extrinsic_filename = "";
    std::string disparity_filename = "";
    std::string point_cloud_filename = "";

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4, STEREO_HH4=5 };
    int alg = STEREO_SGBM;
    int SADWindowSize, numberOfDisparities;
    bool no_display;
    bool color_display;
    float scale;

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
    cv::CommandLineParser parser(argc, argv,
        "{@arg1||}{@arg2||}{help h||}{algorithm||}{max-disparity|0|}{blocksize|0|}{no-display||}{color||}{scale|1|}{i||}{e||}{o||}{p||}");
    if(parser.has("help"))
    {
        print_help(argv);
        return 0;
    }
    img1_filename = samples::findFile(parser.get<std::string>(0));
    img2_filename = samples::findFile(parser.get<std::string>(1));
    if (parser.has("algorithm"))
    {
        std::string _alg = parser.get<std::string>("algorithm");
        alg = 
            _alg == "sgbm" ? STEREO_SGBM :
            _alg == "hh" ? STEREO_HH :
            _alg == "var" ? STEREO_VAR :
            _alg == "hh4" ? STEREO_HH4 :
            _alg == "sgbm3way" ? STEREO_3WAY : -1;
    }
    numberOfDisparities = parser.get<int>("max-disparity");
    SADWindowSize = parser.get<int>("blocksize");
    scale = parser.get<float>("scale");
    no_display = parser.has("no-display");
    color_display = parser.has("color");
    if( parser.has("i") )
        intrinsic_filename = parser.get<std::string>("i");
    if( parser.has("e") )
        extrinsic_filename = parser.get<std::string>("e");
    if( parser.has("o") )
        disparity_filename = parser.get<std::string>("o");
    if( parser.has("p") )
        point_cloud_filename = parser.get<std::string>("p");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    if( alg < 0 )
    {
        printf("Command-line parameter error: Unknown stereo algorithm\n\n");
        print_help(argv);
        return -1;
    }
    if ( numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
    {
        printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
        print_help(argv);
        return -1;
    }
    if (scale < 0)
    {
        printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
        return -1;
    }
    if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
    {
        printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
        return -1;
    }
    if( img1_filename.empty() || img2_filename.empty() )
    {
        printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }
    if( (!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()) )
    {
        printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
        return -1;
    }

    if( extrinsic_filename.empty() && !point_cloud_filename.empty() )
    {
        printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
        return -1;
    }

    int color_mode = -1;
    Mat img1 = imread(img1_filename, color_mode);
    
    Mat img2 = imread(img2_filename, color_mode);

    imshow("1", img1);

    if (img1.empty())
    {
        printf("Command-line parameter error: could not load the first input image file\n");
        return -1;
    }
    if (img2.empty())
    {
        printf("Command-line parameter error: could not load the second input image file\n");
        return -1;
    }

    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();
    

    Rect roi1, roi2;
    Mat Q;

    Mat M1, T;
    if( !intrinsic_filename.empty() )
    {
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename.c_str());
            return -1;
        }

        Mat D1, D2, M2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        fs.open(extrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename.c_str());
            return -1;
        }

        Mat R, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;

        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;
        
    }
    
    cout<<endl<<img_size<<endl;

    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);// 9

    int cn = img1.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize); // 8*9*9
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);// 32*9*9
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities); //96
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    if(alg==STEREO_HH)
        sgbm->setMode(StereoSGBM::MODE_HH);
    else if(alg==STEREO_SGBM)
        sgbm->setMode(StereoSGBM::MODE_SGBM);
    else if(alg==STEREO_HH4)
        sgbm->setMode(StereoSGBM::MODE_HH4);
    else if(alg==STEREO_3WAY)
        sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

    Mat disp, disp8;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

    int64 t = getTickCount();
    float disparity_multiplier = 1.0f;
    
    if( alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_HH4 || alg == STEREO_3WAY )
    {
        sgbm->compute(img1, img2, disp);
        if (disp.type() == CV_16S)
            disparity_multiplier = 16.0f;
    }
    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if( alg != STEREO_VAR )
        disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    else
        disp.convertTo(disp8, CV_8U);

    Mat disp8_3c;
    if (color_display)
        cv::applyColorMap(disp8, disp8_3c, COLORMAP_TURBO);

    // if(!disparity_filename.empty())
    //     imwrite(disparity_filename, color_display ? disp8_3c : disp8);

    // cout<< img1.rows<<endl;
    // cout<< img1.cols<<endl;

    // cout <<img1.channels()<<endl;
    imshow("disp", color_display ? disp8_3c : disp8);
    imshow("left_image", img1);
    imshow("right_image", img2);


    cout<<M1.rows<<endl;
    cout<<M1.cols<<endl;
    cout<<M1.channels()<<endl;
    // cout<<M1.type()<<endl;
    double fx = M1.at<double>(0,0), fy = M1.at<double>(1,1);
    double u0 = M1.at<double>(0,2), v0 = M1.at<double>(1,2);
    
    double baseline = cv::norm(T);
    //cout<<fx<<' '<<fy<<' '<<u0<<' '<<v0<<' '<<baseline<<endl;

    // cv::resize(left, left, cv::Size(640,550));
    // cv::resize(right, right, cv::Size(640,550));

    cv::Mat floatDisp;
    disp.convertTo(floatDisp, CV_32F, 1.0 / disparity_multiplier);
    

    Disp2PointCloud(
    floatDisp, img1, img2, 
    fx, fy, 
    u0, v0, baseline);


    return 0;
}




void showPointCloud(const vector<Vector3d, Eigen::aligned_allocator<Vector3d>> &pointcloud_pos,
    const vector<Vector3d, Eigen::aligned_allocator<Vector3d>> &pointcloud_bgr) 
{

    if (pointcloud_pos.empty() || pointcloud_bgr.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        // for (auto &p: pointcloud_pos) {
        //     glColor3f(p[3], p[4], p[5]);
        //     glVertex3d(p[0], p[1], p[2]);
        // }
        for (auto it = pointcloud_pos.begin(); it != pointcloud_pos.end(); ++it)
        {
            const Vector3d &p = *it;
            const Vector3d &color = pointcloud_bgr[std::distance(pointcloud_pos.begin(), it)];

            glColor3f(color[0], color[1], color[2]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}



void Disp2PointCloud(
    const cv::Mat disparity, const cv::Mat left, const cv::Mat right, 
    double fx, double fy, 
    double u0, double v0, double baseline)
{
    // 生成点云
    vector<Vector3d, Eigen::aligned_allocator<Vector3d>> pointcloud_pos;
    vector<Vector3d, Eigen::aligned_allocator<Vector3d>> pointcloud_bgr;

    int cn = left.channels();
    for (int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++) {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) {continue;cout<<"1";};

            Vector3d point_pos(0, 0, 0);
            Vector3d point_bgr(0, 0, 0);
            if (cn==1)
            {
                point_bgr[0] = left.at<uchar>(v, u) / 255.0;
                point_bgr[1] = left.at<uchar>(v, u) / 255.0;
                point_bgr[2] = left.at<uchar>(v, u) / 255.0;
            }
            else
            {
                point_bgr[0] = left.at<uchar>(v, u, 0) / 255.0;
                point_bgr[1] = left.at<uchar>(v, u, 1) / 255.0;
                point_bgr[2] = left.at<uchar>(v, u, 2) / 255.0;
            }
            // 根据双目模型计算 point 的位置
            double x = (u - u0) / fx;
            double y = (v - v0) / fy;
            double depth = fx * baseline / (disparity.at<float>(v, u));//Z = f * b / d, 其中 d = u_l - u_r
            point_pos[0] = x * depth;//X = (u - cx) / fx * Z
            point_pos[1] = y * depth;//Y = (v - cy) / fy * Z
            point_pos[2] = depth;

            pointcloud_pos.push_back(point_pos);
            pointcloud_bgr.push_back(point_bgr);
        }

    // cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);
    // 画出点云
    showPointCloud(pointcloud_pos, pointcloud_bgr);

}