#include <stdio.h>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>

using namespace std;

const int width = 640;
const int height = 480;
const double f_x = width; // / 2 * 2.25 / 1.1917536;
//1920 / 2 / 2.25;
//0.8391 * width;

vector<cv::Mat> imgs;
vector<shared_ptr<open3d::geometry::PointCloud>> pcd_ptrs;
cv::Mat reprojected;
cv::Mat id_img;

int dataNo = 0;
vector<int> data_ids = {700, 1290, 1460, 2350, 3850}; //1100 // 2/4 miyanosawa
//vector<int> data_ids = {10, 20, 30, 40, 50}; // 2/19 13jo

/*
    for (int i = 0; i < 10; i += 1)
    {
        data_ids.emplace_back(i);
    }
    */

// 02_04_miyanosawa

string folder_path = "../../../data/2020_02_04_miyanosawa/";
int X = 506;
int Y = 483;
int Z = 495;
int roll = 568;
int pitch = 551;
int yaw = 510;

// 03_03_miyanosawa
/*
int X = 500;
int Y = 474;
int Z = 458;
int theta = 506;
int phi = 527;
*/

int u0 = 0;
int v0 = 0;
int rate = 1;

void reproject()
{
    cv::Mat thermal_img = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat points_img = cv::Mat::zeros(height, width, CV_8UC3);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (v0 <= i && i < v0 + height / rate && u0 <= j && j < u0 + width / rate)
            {
                for (int k1 = 0; k1 < rate; k1++)
                {
                    for (int k2 = 0; k2 < rate; k2++)
                    {
                        reprojected.at<cv::Vec3b>((i - v0) * rate + k1, (j - u0) * rate + k2) = imgs[dataNo].at<cv::Vec3b>(i, j);
                        thermal_img.at<cv::Vec3b>((i - v0) * rate + k1, (j - u0) * rate + k2) = imgs[dataNo].at<cv::Vec3b>(i, j);
                    }
                }
            }
            id_img.at<unsigned short>(i, j) = 0;
        }
    }
    for (int i = 0; i < pcd_ptrs[dataNo]->points_.size(); i++)
    {

        double rawX = pcd_ptrs[dataNo]->points_[i][0];
        double rawY = pcd_ptrs[dataNo]->points_[i][1];
        double rawZ = pcd_ptrs[dataNo]->points_[i][2];

        double r = sqrt(rawX * rawX + rawZ * rawZ);
        double rollVal = (roll - 500) / 1000.0;
        double pitchVal = (pitch - 500) / 1000.0;
        double yawVal = (yaw - 500) / 1000.0;
        double xp = cos(yawVal) * cos(pitchVal) * rawX + (cos(yawVal) * sin(pitchVal) * sin(rollVal) - sin(yawVal) * cos(rollVal)) * rawY + (cos(yawVal) * sin(pitchVal) * cos(rollVal) + sin(yawVal) * sin(rollVal)) * rawZ;
        double yp = sin(yawVal) * cos(pitchVal) * rawX + (sin(yawVal) * sin(pitchVal) * sin(rollVal) + cos(yawVal) * cos(rollVal)) * rawY + (sin(yawVal) * sin(pitchVal) * cos(rollVal) - cos(yawVal) * sin(rollVal)) * rawZ;
        double zp = -sin(pitchVal) * rawX + cos(pitchVal) * sin(rollVal) * rawY + cos(pitchVal) * cos(rollVal) * rawZ;
        double x = xp + (X - 500) / 100.0;
        double y = yp + (Y - 500) / 100.0;
        double z = zp + (Z - 500) / 100.0;

        if (z > 0)
        {
            int u = (int)(width / 2 + f_x * x / z);
            int v = (int)(height / 2 + f_x * y / z);
            if (0 <= u && u < width && 0 <= v && v < height)
            {
                id_img.at<unsigned short>(v, u) = i + 1;
                if (v0 <= v && v < v0 + height / rate && u0 <= u && u < u0 + width / rate)
                {
                    int color = (int)(z * 1000);
                    for (int k1 = 0; k1 < rate; k1++)
                    {
                        for (int k2 = 0; k2 < rate; k2++)
                        {
                            reprojected.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[0] = color % 255;
                            reprojected.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[1] = color / 255 % 255;
                            reprojected.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[2] = color / 255 / 255 % 255;
                            points_img.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[0] = color % 255;
                            points_img.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[1] = color / 255 % 255;
                            points_img.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[2] = color / 255 / 255 % 255;
                        }
                    }
                }
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
void on_trackbarRoll(int val, void *object)
{
    roll = val;
    reproject();
}
void on_trackbarPitch(int val, void *object)
{
    pitch = val;
    reproject();
}
void on_trackbarYaw(int val, void *object)
{
    yaw = val;
    reproject();
}
void on_trackbarU0(int val, void *object)
{
    u0 = val;
    reproject();
}
void on_trackbarV0(int val, void *object)
{
    v0 = val;
    reproject();
}

int main(int argc, char *argv[])
{

    vector<double> tans;
    double PI = acos(-1);
    double rad = (-16.6 + 0.26349) * PI / 180;
    double delta_rad = 0.52698 * PI / 180;
    double max_rad = (16.6 + 0.26349) * PI / 180;
    while (rad < max_rad + 0.00001)
    {
        tans.emplace_back(tan(rad));
        rad += delta_rad;
    }
    int layers = 64;

    for (int i = 0; i < data_ids.size(); i++)
    {
        string img_path = folder_path + to_string(data_ids[i]) + "_rgb.png";
        imgs.emplace_back(cv::imread(img_path));

        string pcd_path = folder_path + to_string(data_ids[i]) + ".pcd";
        open3d::geometry::PointCloud pointcloud;
        auto pcd_ptr = make_shared<open3d::geometry::PointCloud>();
        if (!open3d::io::ReadPointCloud(pcd_path, pointcloud))
        {
            cout << "Cannot read" << endl;
        }
        for (int i = 0; i < pointcloud.points_.size(); i++)
        {
            double x = pointcloud.points_[i][1];
            double y = -pointcloud.points_[i][2];
            double z = -pointcloud.points_[i][0];

            double r = sqrt(x * x + z * z);
            auto it = lower_bound(tans.begin(), tans.end(), y / r);
            int index = it - tans.begin();
            if (index % (64 / layers) == 0)
            {
                pcd_ptr->points_.emplace_back(x, y, z);
            }
        }
        pcd_ptrs.emplace_back(pcd_ptr);
    }

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Data No", "Image", &dataNo, data_ids.size() - 1, &on_trackbarDataNo);
    if (rate > 1)
    {
        cv::createTrackbar("U0", "Image", &u0, width - width / rate, &on_trackbarU0);
        cv::createTrackbar("V0", "Image", &v0, height - height / rate, &on_trackbarV0);
    }
    cv::createTrackbar("X(-5,5)", "Image", &X, 1000, &on_trackbarX);
    cv::createTrackbar("Y(-5,5)", "Image", &Y, 1000, &on_trackbarY);
    cv::createTrackbar("Z(-5,5)", "Image", &Z, 1000, &on_trackbarZ);
    cv::createTrackbar("Roll(-1,1)", "Image", &roll, 1000, &on_trackbarRoll);
    cv::createTrackbar("Pitch(-1,1)", "Image", &pitch, 1000, &on_trackbarPitch);
    cv::createTrackbar("Yaw(-1,1)", "Image", &yaw, 1000, &on_trackbarYaw);

    id_img = cv::Mat::zeros(height, width, CV_16SC1);
    reprojected = cv::Mat::zeros(height, width, CV_8UC3);
    reproject();

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}