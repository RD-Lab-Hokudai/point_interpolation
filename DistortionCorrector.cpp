#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

/*
double k1 = 1.05791597e-06;
double k2 = 5.26154073e-14;
double k3 = 3.41991153e-06;
double k4 = 3.27612688e-13;
double p1 = -4.30326545e-06;
double p2 = -4.60648477e-06;
double alpha = 1.0;
int width = 882;
int height = 560;
*/

// Consider y coefficient
double k1 = 1.21456239e-05;
double k2 = 1.96249030e-14;
double k3 = 1.65216912e-05;
double k4 = 1.53712821e-11;
double p1 = -2.42560758e-06;
double p2 = -4.05806821e-06;
double alpha = 9.88930697e-01;
//int width = 882;
//int height = 560;
// Ignore p
int width = 938;
int height = 606;

int main(int argc, char *argv[])
{
    cv::Mat input = cv::imread("../distorted.png");

    cv::Mat img = cv::Mat::zeros(1200, 1700, CV_8UC3);
    for (int i = 0; i < img.rows; i++)
    {
        cv::Vec3b *out = img.ptr<cv::Vec3b>(i);
        for (int j = 0; j < img.cols; j++)
        {
            double x1 = j - img.cols / 2;
            double y1 = (img.rows / 2 - i) * alpha;
            double r_2 = x1 * x1 + y1 * y1;
            double r_4 = r_2 * r_2;
            double x2 = x1 * (1 + k1 * r_2 + k2 * r_4) / (1 + k3 * r_2 + k4 * r_4) + 2 * p1 * x1 * y1 + p2 * (r_2 + 2 * x1 * x1);
            double y2 = y1 * (1 + k1 * r_2 + k2 * r_4) / (1 + k3 * r_2 + k4 * r_4) + 2 * p2 * x1 * y1 + p1 * (r_2 + 2 * y1 * y1);
            int ii = input.rows / 2 - (int)round(y2);
            int jj = input.cols / 2 + (int)round(x2);
            if (0 <= ii && ii < input.rows && 0 <= jj && jj < input.cols)
            {
                out[j] = input.at<cv::Vec3b>(ii, jj);
            }
        }
    }

    cv::imwrite("../corrected.png", img);
    cv::imshow("B", img);
    cv::waitKey();
}