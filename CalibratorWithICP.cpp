#include <stdio.h>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>

using namespace std;

//const int width = 882;
//const int height = 560;
const int width = 938;
const int height = 606;
//const int width = 672;
//const int height = 376;
const double f_x = width / 2 * 1.01;

vector<cv::Mat> imgs;
vector<shared_ptr<open3d::geometry::PointCloud>> pcd_ptrs;
cv::Mat reprojected;

int dataNo = 0;

// 02_19_13jo
/*
int X = 495;
int Y = 485;
int Z = 509;
int roll = 481;
int pitch = 524;
int yaw = 502;
*/
// 02_04_miyanosawa

int X = 495;
int Y = 475;
int Z = 458;
int roll = 488;
int pitch = 568;
int yaw = 500;

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
    cv::Mat reprojected2 = cv::Mat::zeros(height, width, CV_8UC3);
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
                        reprojected2.at<cv::Vec3b>((i - v0) * rate + k1, (j - u0) * rate + k2) = imgs[dataNo].at<cv::Vec3b>(i, j);
                        thermal_img.at<cv::Vec3b>((i - v0) * rate + k1, (j - u0) * rate + k2) = imgs[dataNo].at<cv::Vec3b>(i, j);
                    }
                }
            }
        }
    }

    vector<tuple<int, int>> uvs(pcd_ptrs[dataNo]->points_.size());
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
            uvs[i] = make_tuple(u, v);
            if (0 <= u && u < width && 0 <= v && v < height)
            {
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

    {
        auto points_pcd = make_shared<open3d::geometry::PointCloud>();
        auto img_pcd = make_shared<open3d::geometry::PointCloud>();

        vector<int> is_edges(pcd_ptrs[dataNo]->points_.size(), 0);
        cv::Mat point_edge_img = cv::Mat::zeros(height, width, CV_8UC1);
        for (int i = 1; i < pcd_ptrs[dataNo]->points_.size(); i++)
        {
            while (i + 1 < pcd_ptrs[dataNo]->points_.size() && pcd_ptrs[dataNo]->points_[i][0] < pcd_ptrs[dataNo]->points_[i + 1][0])
            {
                double l1 = (pcd_ptrs[dataNo]->points_[i] - pcd_ptrs[dataNo]->points_[i - 1]).norm();
                double l2 = (pcd_ptrs[dataNo]->points_[i + 1] - pcd_ptrs[dataNo]->points_[i]).norm();
                if (max(l1, l2) / min(l1, l2) > 4)
                {
                    is_edges[i] = 2;
                }
                i++;
            }
        }
        for (int i = 0; i < pcd_ptrs[dataNo]->points_.size(); i++)
        {
            if (i - 1 >= 0 && i + 1 < pcd_ptrs[dataNo]->points_.size() && is_edges[i - 1] == 2 && is_edges[i + 1] == 2)
            {
                is_edges[i] = 1;
            }
            if (is_edges[i] > 0)
            {
                int u = get<0>(uvs[i]);
                int v = get<1>(uvs[i]);
                if (0 <= u && u < width && 0 <= v && v < height)
                {
                    point_edge_img.at<uchar>(v, u) = 255;
                }
            }
        }

        cv::Mat img_edges = cv::Mat::zeros(height, width, CV_8UC1);
        //vector<vector<int>> is_edges_img(height, vector<int>(width, 0));
        for (int i = 0; i < height; i++)
        {
            for (int j = 1; j + 1 < width; j++)
            {
                double l1 = imgs[dataNo].at<cv::Vec3b>(i, j)[0] - imgs[dataNo].at<cv::Vec3b>(i, j - 1)[0];
                double l2 = imgs[dataNo].at<cv::Vec3b>(i, j + 1)[0] - imgs[dataNo].at<cv::Vec3b>(i, j)[0];
                if (max(l1, l2) / min(l1, l2) > 100)
                {
                    img_edges.at<uchar>(i, j) = 255;
                }
            }
        }
        cv::Canny(imgs[dataNo], img_edges, 30, 50);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (point_edge_img.at<uchar>(i, j) > 0)
                {
                    points_pcd->points_.emplace_back(j, i, 0);
                    points_pcd->colors_.emplace_back(1, 0, 0);
                }
                if (img_edges.at<uchar>(i, j) > 0)
                {
                    img_pcd->points_.emplace_back(j, i, 0);
                    img_pcd->colors_.emplace_back(0, 1, 0);
                }
            }
        }

        auto res_icp = open3d::registration::RegistrationICP(*points_pcd, *img_pcd, 5);
        cout << res_icp.transformation_ << endl;
        cout << res_icp.fitness_ << endl;
        open3d::visualization::DrawGeometries({points_pcd, img_pcd}, "PointCloud", 1600, 900);
        points_pcd->Transform(res_icp.transformation_);
        open3d::visualization::DrawGeometries({points_pcd, img_pcd}, "PointCloud", 1600, 900);
        for (int i = 0; i < points_pcd->points_.size(); i++)
        {
            int u = points_pcd->points_[i][0];
            int v = points_pcd->points_[i][1];
            if (v0 <= v && v < v0 + height / rate && u0 <= u && u < u0 + width / rate)
            {
                for (int k1 = 0; k1 < rate; k1++)
                {
                    for (int k2 = 0; k2 < rate; k2++)
                    {
                        reprojected2.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[0] = 255;
                        reprojected2.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[1] = 255;
                        reprojected2.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[2] = 255;
                    }
                }
            }
        }
        cv::imshow("A", point_edge_img);
        cv::imshow("B", img_edges);
        cv::imshow("C", reprojected2);
        cv::waitKey();
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
    vector<int> data_ids = {700, 1290, 1460, 2350, 3850}; //1100 // 2/4 miyanosawa
    //vector<int> data_ids = {10, 20, 30, 40, 50}; // 2/19 13jo

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
    int layer_cnt = 16;

    for (int i = 0; i < data_ids.size(); i++)
    {
        string img_path = "../../../data/2020_02_04_miyanosawa/" + to_string(data_ids[i]) + ".png";
        imgs.emplace_back(cv::imread(img_path));

        string pcd_path = "../../../data/2020_02_04_miyanosawa/" + to_string(data_ids[i]) + ".pcd";
        open3d::geometry::PointCloud pointcloud;
        auto pcd_ptr = make_shared<open3d::geometry::PointCloud>();
        vector<vector<Eigen::Vector3d>> layers(layer_cnt, vector<Eigen::Vector3d>());
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
            if (index % (64 / layer_cnt) == 0)
            {
                layers[index / (64 / layer_cnt)].emplace_back(x, y, z);
            }
        }
        for (int i = 0; i < layer_cnt; i++)
        {
            for (int j = 0; j < layers[i].size(); j++)
            {
                pcd_ptr->points_.emplace_back(layers[i][j]);
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

    reprojected = cv::Mat::zeros(height, width, CV_8UC3);
    reproject();

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}