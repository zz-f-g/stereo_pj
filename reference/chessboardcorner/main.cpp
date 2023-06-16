#include <opencv2/opencv.hpp>
#include <istream>
#include <stdint.h>
#include <queue>

#define SCORE_T uint16_t

class imgPoint
{
public:
    cv::Point point;
    SCORE_T intensity;
    bool operator>(const imgPoint &p) const
    {
        return this->intensity > p.intensity;
    }
};

void findTopN(const cv::Mat &matrix, uint32_t n, std::vector<cv::Point> &topNPoints)
{
    topNPoints.resize(0);
    std::priority_queue<imgPoint, std::vector<imgPoint>, std::greater<imgPoint>> minHeap;
    imgPoint p;

    for (int row = 0; row < matrix.rows; row++)
    {
        for (int col = 0; col < matrix.cols; col++)
        {
            p.point = cv::Point(col, row);
            p.intensity = matrix.at<uint16_t>(row, col);

            if (minHeap.size() < n)
            {
                // 堆未满时直接插入元素
                minHeap.push(p);
            }
            else
            {
                // 堆已满时比较当前值与堆底元素大小
                if (p > minHeap.top())
                {
                    minHeap.pop();
                    minHeap.push(p);
                }
            }
        }
    }

    // 从堆中提取前n个最大值
    while (!minHeap.empty())
    {
        p = minHeap.top();
        minHeap.pop();
        topNPoints.push_back(p.point);
    }
}

void nonMaximumSuppression(cv::Mat &input, int windowSize)
{
    // 确保输入矩阵是单通道灰度图像
    CV_Assert(input.channels() == 1);

    int width = input.cols;
    int height = input.rows;

    int halfWindow = windowSize / 2;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if ((y < halfWindow) || (y >= height - halfWindow) || (x < halfWindow) || (x >= width - halfWindow))
            {
                input.at<SCORE_T>(y, x) = 0;
                continue;
            }
            SCORE_T centerValue = input.at<SCORE_T>(y, x);
            bool isMaximum = true;

            // 在窗口范围内比较中心像素与周围像素的值
            for (int i = -halfWindow; i <= halfWindow; i++)
            {
                for (int j = -halfWindow; j <= halfWindow; j++)
                {
                    SCORE_T neighborValue = input.at<SCORE_T>(y + i, x + j);
                    if ((i == 0) && (j == 0))
                    {
                        continue;
                    }
                    if (neighborValue >= centerValue)
                    {
                        input.at<SCORE_T>(y, x) = 0;
                        isMaximum = false;
                        break;
                    }
                }
                if (!isMaximum)
                {
                    break;
                }
            }
        }
    }
}

namespace calib
{
    bool findChessboardCorners(const cv::Mat &chessboardimage,
                               cv::Size boardsize,
                               std::vector<cv::Point2i> &corners,
                               uint32_t blockradius = 8);
};

bool calib::findChessboardCorners(const cv::Mat &chessboardimage,
                                      cv::Size boardsize,
                                      std::vector<cv::Point2i> &corners,
                                      uint32_t blockradius)
{
    cv::Mat grayimg, binimg;
    cv::cvtColor(chessboardimage, grayimg, cv::COLOR_BGR2GRAY);
    cv::threshold(grayimg, binimg, 0, 255, cv::THRESH_OTSU);
    cv::Size imgsize = chessboardimage.size();
    /*
    cv::namedWindow("binary");
    cv::imshow("binary", binimg);
    cv::waitKey();
    */

    cv::Mat originBlock, rotatedBlock;
    cv::Mat score;
    score = cv::Mat::zeros(imgsize, CV_16UC1); // SCORE_T
    cv::Rect roi;

    for (uint32_t i = blockradius; i < imgsize.height - blockradius; i++)
    {
        for (uint32_t j = blockradius; j < imgsize.width - blockradius; j++)
        {
            roi = cv::Rect(j - blockradius, i - blockradius, 2 * blockradius + 1, 2 * blockradius + 1);
            originBlock = binimg(roi);
            cv::rotate(originBlock, rotatedBlock, cv::ROTATE_90_CLOCKWISE);

            SCORE_T s = 0;
            for (uint32_t k = 0; k < (2 * blockradius + 1) * (2 * blockradius + 1); k++)
            {
                if (originBlock.at<uchar>(k / (2 * blockradius + 1), k % (2 * blockradius + 1)) ^ rotatedBlock.at<uchar>(k / (2 * blockradius + 1), k % (2 * blockradius + 1)))
                {
                    s += 1;
                }
            }
            score.at<SCORE_T>(i, j) = s;
            // std::cout << i << ", " << j << std::endl;
            // std::cout << "score = " << s << std::endl;
            // cv::waitKey();
        }
    }
    nonMaximumSuppression(score, 4);

    cv::Mat score_8u;
    score.convertTo(score_8u, CV_8UC1);
    cv::namedWindow("score");
    cv::imshow("score", score_8u);
    cv::waitKey();
    //cv::imshow("score", score);

    findTopN(score, boardsize.width * boardsize.height, corners);
    std::cout << corners << std::endl;
    cv::Mat bg = cv::Mat::zeros(score.size(), CV_8UC3);
    for (int32_t i = 0; i < boardsize.width * boardsize.height; i++)
    {
        cv::circle(bg, corners.at(i), 5, cv::Scalar(0, 0, 255), -1); // 绘制圆圈
    }

    cv::namedWindow("corners");
    cv::imshow("corners", bg);
    cv::waitKey();
    return true;
}

int main(int argc, char **argv)
{
    std::string imgname = "chessboard.jpg";
    if (2 == argc)
    {
        imgname = argv[1];
    }
    cv::Mat img;
    img = cv::imread(imgname);

    cv::Size boardsize = cv::Size(9, 6);

    std::vector<cv::Point2i> corners;
    calib::findChessboardCorners(img, boardsize, corners);

    cv::destroyAllWindows();

    /*
    cv::Mat score = cv::Mat::eye(3, 3, CV_16UC1);
    std::cout << score << std::endl;
    std::vector<cv::Point>corners;
    findTopN(score, 3, corners);
    std::cout << corners << std::endl;
    */

    return 0;
}
