#include <stdio.h>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>

using namespace std;

//const int width = 882;
//const int height = 560;
const int width = 938;
const int height = 606;
const double f_x = width / 2 * 1.01;

vector<cv::Mat> imgs;
vector<shared_ptr<open3d::geometry::PointCloud>> pcd_ptrs;
cv::Mat reprojected;

int dataNo = 0;
int X = 495;
int Y = 499;
int Z = 499;
int theta = 502;

void reproject()
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            reprojected.at<cv::Vec3b>(i, j) = imgs[dataNo].at<cv::Vec3b>(i, j);
        }
    }
    for (int i = 0; i < pcd_ptrs[dataNo]->points_.size(); i++)
    {
        double rawX = pcd_ptrs[dataNo]->points_[i][0];
        double rawY = pcd_ptrs[dataNo]->points_[i][1];
        double rawZ = pcd_ptrs[dataNo]->points_[i][2];

        double xp = rawX * cos((theta - 500) / 1000.0) - rawZ * sin((theta - 500) / 1000.0);
        double yp = rawY;
        double zp = rawX * sin((theta - 500) / 1000.0) + rawZ * cos((theta - 500) / 1000.0);
        double x = xp + (X - 500) / 100.0;
        double y = yp + (Y - 500) / 100.0;
        double z = zp + (Z - 500) / 100.0;

        if (z > 0)
        {
            int u = (int)(width / 2 + f_x * x / z);
            int v = (int)(height / 2 + f_x * y / z);
            if (0 <= u && u < width && 0 <= v && v < height)
            {
                int color = (int)(z * 1000);
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
    vector<int> data_ids = {550, 1000, 1125, 1260, 1550};
    for (int i = 0; i < data_ids.size(); i++)
    {
        string img_path = "../" + to_string(data_ids[i]) + ".png";
        imgs.emplace_back(cv::imread(img_path));
        if (i == 0)
        {
            reprojected = cv::imread(img_path);
        }

        string pcd_path = "../" + to_string(data_ids[i]) + ".pcd";
        open3d::geometry::PointCloud pointcloud;
        if (!open3d::io::ReadPointCloud(pcd_path, pointcloud))
        {
            cout << "Cannot read" << endl;
        }
        auto pcd_ptr = make_shared<open3d::geometry::PointCloud>(pointcloud);
        for (int i = 0; i < pcd_ptr->points_.size(); i++)
        {
            double x = pcd_ptr->points_[i][1];
            double y = -pcd_ptr->points_[i][2];
            double z = -pcd_ptr->points_[i][0];
            pcd_ptr->points_[i] = Eigen::Vector3d(x, y, z);
        }
        pcd_ptrs.emplace_back(pcd_ptr);
    }

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Data No", "Image", &dataNo, data_ids.size() - 1, &on_trackbarDataNo);
    cv::createTrackbar("X(-1,1)", "Image", &X, 1000, &on_trackbarX);
    cv::createTrackbar("Y(-1,1)", "Image", &Y, 1000, &on_trackbarY);
    cv::createTrackbar("Z(-1,1)", "Image", &Z, 1000, &on_trackbarZ);
    cv::createTrackbar("theta(-1,1)", "Image", &theta, 1000, &on_trackbarTheta);
    cv::imshow("Image", reprojected);
    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}