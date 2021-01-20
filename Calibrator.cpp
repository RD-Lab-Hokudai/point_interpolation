#include <stdio.h>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>

using namespace std;

//const int width = 882;
//const int height = 560;
const int width = 938;
const int height = 606;
//const int width = 672;
//const int height = 376;
const double f_x = width / 2 * 1.01;

vector<cv::Mat> imgs;
vector<shared_ptr<open3d::geometry::PointCloud> > pcd_ptrs;
cv::Mat reprojected;

int dataNo = 0;

// 02_19_13jo
/*
vector<int> data_ids = {10, 50, 100, 150, 200};
int X = 502;
int Y = 484;
int Z = 499;
int roll = 478;
int pitch = 520;
int yaw = 502;
*/

// 02_04_miyanosawa
/*
vector<int> data_ids = {700, 1290, 1460, 2350, 3850}; //1100 
int X = 495;
int Y = 475;
int Z = 458;
int roll = 488;
int pitch = 568;
int yaw = 500;
*/
// 03_03_miyanosawa
/*
int X = 495;
int Y = 466;
int Z = 450;
int roll = 469;
int pitch = 503;
int yaw = 487;
*/

// 2021_01_15_teine
string folder_path = "../../../data/2021_01_15_teine/";
vector<int> data_ids = {100, 200, 300, 400, 500};
int X = 520;
int Y = 499;
int Z = 499;
int roll = 527;
int pitch = 3700;
int yaw = 527;

void reproject()
{
    cv::Mat thermal_img = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat points_img = cv::Mat::zeros(height, width, CV_8UC3);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            reprojected.at<cv::Vec3b>(i, j) = imgs[dataNo].at<cv::Vec3b>(i, j);
            thermal_img.at<cv::Vec3b>(i, j) = imgs[dataNo].at<cv::Vec3b>(i, j);
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
                uchar color = (uchar)min(z * 100, 255.0);
                cv::circle(reprojected, cv::Point(u, v), 1, cv::Scalar(255, 255, 0));
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

int main(int argc, char *argv[])
{

    /*
    for (int i = 0; i < 10; i += 1)
    {
        data_ids.emplace_back(i);
    }
    */

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
        string img_path = folder_path + to_string(data_ids[i]) + ".png";
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
            if (true || index % (64 / layers) == 0)
            {
                pcd_ptr->points_.emplace_back(x, y, z);
            }
        }
        //open3d::visualization::DrawGeometries({pcd_ptr});
        pcd_ptrs.emplace_back(pcd_ptr);
    }

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Data No", "Image", &dataNo, data_ids.size() - 1, &on_trackbarDataNo);
    cv::createTrackbar("X(-5,5)", "Image", &X, 1000, &on_trackbarX);
    cv::createTrackbar("Y(-5,5)", "Image", &Y, 1000, &on_trackbarY);
    cv::createTrackbar("Z(-5,5)", "Image", &Z, 1000, &on_trackbarZ);
    cv::createTrackbar("Roll(-1,1)", "Image", &roll, 10000, &on_trackbarRoll);
    cv::createTrackbar("Pitch(-1,1)", "Image", &pitch, 10000, &on_trackbarPitch);
    cv::createTrackbar("Yaw(-1,1)", "Image", &yaw, 10000, &on_trackbarYaw);

    reprojected = cv::Mat::zeros(height, width, CV_8UC3);
    reproject();

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}