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

const int width = 640;
const int height = 480;
const int min_points = 1000;

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

shared_ptr<geometry::PointCloud> raw_pcd_ptr;
cv::Mat img;

Eigen::VectorXd calc_filtered(const Eigen::VectorXd &params, bool see_res = false)
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
    double f_x = params[3];
    for (int i = 0; i < raw_pcd_ptr->points_.size(); i++)
    {
        double rawX = raw_pcd_ptr->points_[i][1];
        double rawY = -raw_pcd_ptr->points_[i][2];
        double rawZ = -raw_pcd_ptr->points_[i][0];
        double r = sqrt(rawX * rawX + rawZ * rawZ);

        double roll = 0;
        //params[3];
        double pitch = 0;
        //params[4];
        double yaw = 0;
        //params[5];
        double xp = cos(yaw) * cos(pitch) * rawX + (cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll)) * rawY + (cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll)) * rawZ + params[0];
        double yp = sin(yaw) * cos(pitch) * rawX + (sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll)) * rawY + (sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll)) * rawZ + params[1];
        double zp = -sin(pitch) * rawX + cos(pitch) * sin(roll) * rawY + cos(pitch) * cos(roll) * rawZ + params[2];
        double r2 = xp * xp + yp * yp;
        double x = xp;
        //*(1 + params[6] * r2 + params[7] * r2 * r2 + params[8] * r2 * r2 * r2) + 2 * params[9] * xp *yp + params[10] * (r2 + 2 * xp * xp);
        double y = yp;
        //    *(1 + params[6] * r2 + params[7] * r2 * r2 + params[8] * r2 * r2 * r2) + 2 * params[10] * xp *yp + params[9] * (r2 + 2 * yp * yp);
        double z = zp;

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
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            layer_img.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(i, j);
        }
    }

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

    Eigen::VectorXd res = Eigen::VectorXd::Zero(min_points);
    int in_cnt = 0;
    double c = 0.1;
    double r = 10;
    for (int i = 0; i < 64; i++)
    {
        for (size_t j = 0; j + 1 < layers[i].size(); j++)
        {
            if (in_cnt == min_points)
            {
                break;
            }

            double cTmp = c;
            if (is_edges[i][j] > 0 && is_edges[i][j + 1] > 0)
            {
                cTmp = 0;
            }
            int u = (int)(width / 2 + f_x * layers[i][j][0] / layers[i][j][2]);
            int v = (int)(height / 2 + f_x * layers[i][j][1] / layers[i][j][2]);
            int u1 = (int)(width / 2 + f_x * layers[i][j + 1][0] / layers[i][j + 1][2]);
            int v1 = (int)(height / 2 + f_x * layers[i][j + 1][1] / layers[i][j + 1][2]);
            double val = img.at<cv::Vec3b>(v, u)[0] - img.at<cv::Vec3b>(v1, u1)[0];
            double w = r * (1 - exp(-cTmp * val * val));
            res[in_cnt] = w;
            in_cnt++;
        }
    }

    if (in_cnt < min_points)
    {
        cout << "out" << in_cnt << endl;
        for (int i = in_cnt; i < min_points; i++)
        {
            res[i] = 100;
        }
    }

    if (see_res)
    {
        cv::imshow("U", all_layer_img);
        cv::imshow("T", layer_img);
        cv::imshow("S", img);
        cv::waitKey();
    }

    return res;
}

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

struct misra1a_functor : Functor<double>
{
    misra1a_functor(int inputs, int values)
        : inputs_(inputs), values_(values) {}

    int operator()(const Eigen::VectorXd &b, Eigen::VectorXd &fvec) const
    {
        fvec = calc_filtered(b);
        double error = 0;
        for (int i = 0; i < min_points; i++)
        {
            error += abs(fvec[i]);
        }
        cout << error << endl;
        return 0;
    }
    const int inputs_;
    const int values_;
    int inputs() const { return inputs_; }
    int values() const { return values_; }
};

void segmentate(int data_no, bool see_res = false)
{
    const string png_path = "../../../data/2020_02_19_13jo_raw/" + to_string(data_no) + ".png";
    const string pcd_path = "../../../data/2020_02_19_13jo_raw/" + to_string(data_no) + ".pcd";

    geometry::PointCloud pointcloud;
    raw_pcd_ptr = make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(pcd_path, pointcloud))
    {
        cout << "Cannot read" << endl;
    }
    *raw_pcd_ptr = pointcloud;

    img = cv::imread(png_path);

    Eigen::VectorXd params = Eigen::VectorXd::Zero(12);

    params[0] = -0.02;
    params[1] = -0.15;
    params[2] = 0.09;
    params[3] = width / 2;
    /*
    params[4] = -0.1;
    params[5] = 0.17;
    /*
    params[7] = 1.21456239e-05;
    params[8] = 1.96249030e-14;
    params[10] = -2.42560758e-06;
    params[11] = -4.05806821e-06;
    */
    misra1a_functor functor(4, min_points);

    Eigen::NumericalDiff<misra1a_functor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<misra1a_functor>> lm(numDiff);
    lm.minimize(params);
    cout << "check" << endl;
    cout << params << endl;
    calc_filtered(params, true);
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