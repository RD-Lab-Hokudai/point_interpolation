#include <stdio.h>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>

using namespace std;

const int width = 882;
const int height = 560;
const double f_x = width / 2 * 1.01;

cv::Mat img;
cv::Mat reprojected;
shared_ptr<open3d::geometry::PointCloud> pcd_ptr;

int dataNo = 0;
int X = 50;
int Y = 50;
int Z = 50;
int theta = 500;

void reproject()
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            reprojected.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(i, j);
        }
    }
    for (int i = 0; i < pcd_ptr->points_.size(); i++)
    {
        double xp = pcd_ptr->points_[i][0] * cos((theta - 500) / 1000.0) - pcd_ptr->points_[i][2] * sin((theta - 500) / 1000.0);
        double yp = pcd_ptr->points_[i][1];
        double zp = pcd_ptr->points_[i][0] * sin((theta - 500) / 1000.0) + pcd_ptr->points_[i][2] * cos((theta - 500) / 1000.0);
        double x = xp + (X - 50) / 100.0;
        double y = yp + (Y - 50) / 100.0;
        double z = zp + (Z - 50) / 100.0;

        if (z > 0)
        {
            int u = (int)(width / 2 + f_x * x / z);
            int v = (int)(height / 2 + f_x * y / z);
            if (0 <= u && u < width && 0 <= v && v < height)
            {
                int color = z * 100000;
                reprojected.at<cv::Vec3b>(v, u)[0] = color % 255;
                reprojected.at<cv::Vec3b>(v, u)[1] = color / 255 % 255;
                reprojected.at<cv::Vec3b>(v, u)[2] = color / 255 / 255 % 255;
            }
        }
    }
    cv::imshow("Image", reprojected);
}

/* プロトタイプ宣言 */
void on_trackbarDataNo(int val, void *object)
{
    dataNo = val;
    reproject();
}
void on_trackbarX(int val, void *object)
{
    X = val;
    reproject();
}
void on_trackbarY(int val, void *object)
{
    Y = val;
    reproject();
}
void on_trackbarZ(int val, void *object)
{
    Z = val;
    reproject();
}
void on_trackbarTheta(int val, void *object)
{
    theta = val;
    reproject();
}

int main(int argc, char *argv[])
{
    img = cv::imread("../81.png");
    reprojected = cv::imread("../81.png");
    open3d::geometry::PointCloud pointcloud;
    if (!open3d::io::ReadPointCloud("../81.pcd", pointcloud))
    {
        cout << "Cannot read" << endl;
    }
    pcd_ptr = make_shared<open3d::geometry::PointCloud>(pointcloud);
    for (int i = 0; i < pcd_ptr->points_.size(); i++)
    {
        double x = pcd_ptr->points_[i][1];
        double y = -pcd_ptr->points_[i][2];
        double z = -pcd_ptr->points_[i][0];
        pcd_ptr->points_[i] = Eigen::Vector3d(x, y, z);
    }

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("X(-1,1)", "Image", &X, 100, &on_trackbarX);
    cv::createTrackbar("Y(-1,1)", "Image", &Y, 100, &on_trackbarY);
    cv::createTrackbar("Z(-1,1)", "Image", &Z, 100, &on_trackbarZ);
    cv::createTrackbar("theta(-1,1)", "Image", &theta, 1000, &on_trackbarTheta);
    cv::imshow("Image", reprojected);
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}