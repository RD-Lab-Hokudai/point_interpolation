#include <iostream>
#include <vector>
#include <stack>
#include <map>
#include <chrono>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <time.h>

using namespace std;
using namespace open3d;

const int width = 938;
const int height = 606;
//const int width = 882;
//const int height = 560;
const double f_x = width / 2 * 1.01;

// Calibration
// 02_04_13jo
int X = 498;
int Y = 485;
int Z = 509;
int theta = 483;
int phi = 518;
// 02_04_miyanosawa
/*
int X = 495;
int Y = 475;
int Z = 458;
int theta = 438;
int phi = 512;
*/
// 03_03_miyanosawa
/*
int X = 500;
int Y = 474;
int Z = 458;
int theta = 506;
int phi = 527;
*/

// Generic functor
template <typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum
    {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;
};

void calc_filtered(shared_ptr<geometry::PointCloud> raw_pcd_ptr, Eigen::Vector3d params)
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

    vector<vector<Eigen::Vector3d>> all_layers(64, vector<Eigen::Vector3d>());
    for (int i = 0; i < raw_pcd_ptr->points_.size(); i++)
    {
        double rawX = raw_pcd_ptr->points_[i][1];
        double rawY = -raw_pcd_ptr->points_[i][2];
        double rawZ = -raw_pcd_ptr->points_[i][0];

        double r = sqrt(rawX * rawX + rawZ * rawZ);
        double thetaVal = (theta - 500) / 1000.0;
        double phiVal = (phi - 500) / 1000.0;
        double xp = (rawX * cos(phiVal) - rawY * sin(phiVal)) * cos(thetaVal) - (rawZ * cos(phiVal) - rawY * sin(phiVal)) * sin(thetaVal);
        double yp = rawY * cos(phiVal) + r * sin(phiVal);
        double zp = (rawX * cos(phiVal) - rawY * sin(phiVal)) * sin(thetaVal) + (rawZ * cos(phiVal) - rawY * sin(phiVal)) * cos(thetaVal);
        double x = xp + (X - 500) / 100.0;
        double y = yp + (Y - 500) / 100.0;
        double z = zp + (Z - 500) / 100.0;
        /*
        x = rawX;
        y = rawY;
        z = rawZ;
        */

        if (z > 0)
        {
            int u = (int)(width / 2 + f_x * x / z);
            int v = (int)(height / 2 + f_x * y / z);
            if (0 <= u && u < width && 0 <= v && v < height)
            {
                auto it = lower_bound(tans.begin(), tans.end(), rawY / r);
                int index = it - tans.begin();
                all_layers[index].emplace_back(x, y, z);
            }
        }
    }

    vector<vector<Eigen::Vector3d>> layers;
    vector<vector<int>> is_edges;
    cv::Mat all_layer_img = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat layer_img = cv::Mat::zeros(height, width, CV_8UC3);
    for (int i = 0; i < 64; i++)
    {
        // no sort
        vector<Eigen::Vector3d> removed;
        for (size_t j = 0; j < all_layers[i].size(); j++)
        {
            while (removed.size() > 0 && removed.back()[0] * all_layers[i][j][2] >= all_layers[i][j][0] * removed.back()[2])
            {
                removed.pop_back();
            }
            removed.emplace_back(all_layers[i][j]);
        }

        layers.push_back(removed);
        is_edges.push_back({0});

        int u0;
        int v0;
        double l1;
        for (size_t j = 0; j < removed.size(); j++)
        {
            int u = (int)(width / 2 + f_x * removed[j][0] / removed[j][2]);
            int v = (int)(height / 2 + f_x * removed[j][1] / removed[j][2]);
            if (j > 0)
            {
                float yF = v0;
                int x = u0;
                cv::line(all_layer_img, cv::Point(u0, v0), cv::Point(u, v), cv::Scalar(0, 255, 0), 1, 8);

                if (j + 1 < removed.size())
                {
                    double l2 = (removed[j + 1] - removed[j]).norm();
                    double rate = max(l1, l2) / min(l1, l2);
                    if (j == 1 || (j >= 2 && rate < 4))
                    {
                        is_edges[i].emplace_back(0);
                    }
                    else
                    {
                        // # circle(画像, 中心座標, 半径, 色, 線幅, 連結)
                        cv::circle(all_layer_img, cv::Point(u, v), 2, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
                        is_edges[i].emplace_back(2);
                    }
                    l1 = l2;
                    cv::circle(layer_img, cv::Point(u, v), 2, cv::Scalar(255 / rate, 255 * (i % 2), 255), 1, cv::LINE_AA);
                }
            }
            u0 = u;
            v0 = v;
        }

        for (size_t j = 1; j + 1 < is_edges[i].size(); j++)
        {
            if (is_edges[i][j - 1] == 2 && is_edges[i][j + 1] == 2)
            {
                int u = (int)(width / 2 + f_x * removed[j][0] / removed[j][2]);
                int v = (int)(height / 2 + f_x * removed[j][1] / removed[j][2]);
                is_edges[i][j] = 1;
                cv::circle(all_layer_img, cv::Point(u, v), 2, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
            }
        }
    }

    cv::imshow("U", all_layer_img);
    cv::imshow("T", layer_img);
    cv::waitKey();
}

struct misra1a_functor : Functor<double>
{
    misra1a_functor(int values, double *x, double *y, double *z)
        : inputs_(3), values_(values), x(x), y(y), z(z) {}

    double *x;
    double *y;
    double *z;
    int operator()(const Eigen::VectorXd &b, Eigen::VectorXd &fvec) const
    {
        for (int i = 0; i < values_; ++i)
        {
            fvec[i] = b[0] * x[i] + b[1] * y[i] + z[i] - b[2];
        }
        return 0;
    }
    const int inputs_;
    const int values_;
    int inputs() const { return inputs_; }
    int values() const { return values_; }
};

void segmentate(int data_no, bool see_res = false)
{
    const string pcd_path = "../../../data/2020_02_04_13jo/" + to_string(data_no) + ".pcd";
    const bool vertical = true;

    geometry::PointCloud pointcloud;
    auto pcd_ptr = make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(pcd_path, pointcloud))
    {
        cout << "Cannot read" << endl;
    }
    *pcd_ptr = pointcloud;

    calc_filtered(pcd_ptr, Eigen::Vector3d());
}

int main(int argc, char *argv[])
{
    //vector<int> data_nos = {550, 1000, 1125, 1260, 1550}; // 03_03_miyanosawa
    vector<int> data_nos = {10, 20, 30, 40, 50}; // 02_04_13jo
    //vector<int> data_nos = {700, 1290, 1460, 2350, 3850}; // 02_04_miyanosawa
    for (int i = 0; i < data_nos.size(); i++)
    {
        segmentate(data_nos[i], true);
    }
    return 0;
}