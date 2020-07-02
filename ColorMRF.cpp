#include <iostream>
#include <vector>
#include <stack>
#include <map>
#include <chrono>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
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
/*
int X = 498;
int Y = 485;
int Z = 509;
int theta = 483;
int phi = 518;
*/
// 02_04_miyanosawa
int X = 495;
int Y = 475;
int Z = 458;
int theta = 438;
int phi = 512;
// 03_03_miyanosawa
/*
int X = 500;
int Y = 474;
int Z = 458;
int theta = 506;
int phi = 527;
*/

void segmentate(int data_no, int w_trim, bool see_res = false)
{
    string img_path = "../../../data/2020_02_04_miyanosawa/" + to_string(data_no) + ".png";
    string pcd_path = "../../../data/2020_02_04_miyanosawa/" + to_string(data_no) + ".pcd";

    auto img = cv::imread(img_path);
    geometry::PointCloud pointcloud;
    auto pcd_ptr = make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(pcd_path, pointcloud))
    {
        cout << "Cannot read" << endl;
    }
    *pcd_ptr = pointcloud;

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

    int length = width * height;
    vector<cv::Vec3b> params_x(length);
    Eigen::VectorXd params_z(length);
    cv::Mat projected_img = cv::Mat::zeros(height, width, CV_8UC3);

    auto filtered_ptr = make_shared<geometry::PointCloud>();
    vector<vector<double>> base_z(height, vector<double>(width));
    for (int i = 0; i < pcd_ptr->points_.size(); i++)
    {
        double rawX = pcd_ptr->points_[i][1];
        double rawY = -pcd_ptr->points_[i][2];
        double rawZ = -pcd_ptr->points_[i][0];

        double r = sqrt(rawX * rawX + rawZ * rawZ);
        double thetaVal = (theta - 500) / 1000.0;
        double phiVal = (phi - 500) / 1000.0;
        double xp = (rawX * cos(phiVal) - rawY * sin(phiVal)) * cos(thetaVal) - (rawZ * cos(phiVal) - rawY * sin(phiVal)) * sin(thetaVal);
        double yp = rawY * cos(phiVal) + r * sin(phiVal);
        double zp = (rawX * cos(phiVal) - rawY * sin(phiVal)) * sin(thetaVal) + (rawZ * cos(phiVal) - rawY * sin(phiVal)) * cos(thetaVal);
        double x = xp + (X - 500) / 100.0;
        double y = yp + (Y - 500) / 100.0;
        double z = zp + (Z - 500) / 100.0;

        pcd_ptr->points_[i] = Eigen::Vector3d(x, y, z);
        if (pcd_ptr->points_[i][2] > 0)
        {
            int u = (int)(width / 2 + f_x * x / z);
            int v = (int)(height / 2 + f_x * y / z);
            if (0 <= u && u < width && 0 <= v && v < height)
            {
                auto it = lower_bound(tans.begin(), tans.end(), rawY / r);
                int index = it - tans.begin();
                if (index % 4 == 0)
                {
                    filtered_ptr->points_.emplace_back(pcd_ptr->points_[i]);
                    params_z[v * width + u] = pcd_ptr->points_[i][2];
                    projected_img.at<cv::Vec3b>(v, u) = cv::Vec3b(255, 0, 0);
                }
                base_z[v][u] = pcd_ptr->points_[i][2];
            }
        }

        pcd_ptr->points_[i][0] += 100;
    }

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            params_x[i * width + j] = img.at<cv::Vec3b>(i, j);
            if (projected_img.at<cv::Vec3b>(i, j)[0] != 255)
            {
                projected_img.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(i, j);
            }
        }
    }

    //double k = 0.1;
    //double c = 100;
    double k = 0.1;
    double c = 100;
    int h_trim = height;
    int length_trim = w_trim * h_trim;
    Eigen::VectorXd z_trim(length_trim);
    Eigen::SparseMatrix<double> W(length_trim, length_trim);
    vector<Eigen::Triplet<double>> W_triplets;
    for (int i = 0; i < h_trim; i++)
    {
        for (int j = 0; j < w_trim; j++)
        {
            if (params_z[i * width + j] > 0)
            {
                z_trim[i * w_trim + j] = params_z[i * width + j];
                W_triplets.emplace_back(i * w_trim + j, i * w_trim + j, k);
            }
        }
    }
    W.setFromTriplets(W_triplets.begin(), W_triplets.end());

    Eigen::SparseMatrix<double> S(length_trim, length_trim);
    vector<Eigen::Triplet<double>> S_triplets;
    int dires = 4;
    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};
    for (int i = 0; i < h_trim; i++)
    {
        for (int j = 0; j < w_trim; j++)
        {
            double wSum = 0;
            for (int k = 0; k < dires; k++)
            {
                int u = j + dx[k];
                int v = i + dy[k];
                if (0 <= u && u < w_trim && 0 <= v && v < h_trim)
                {
                    double x_norm2 = (params_x[i * width + j][0] - params_x[v * width + u][0]) / 255.0;
                    x_norm2 = x_norm2 * x_norm2;
                    double w = -sqrt(exp(-c * x_norm2));
                    S_triplets.emplace_back(i * w_trim + j, v * w_trim + u, w);
                    wSum += w;
                }
            }
            S_triplets.emplace_back(i * w_trim + j, i * w_trim + j, -wSum);
        }
    }
    S.setFromTriplets(S_triplets.begin(), S_triplets.end());
    Eigen::SparseMatrix<double> A = S.transpose() * S + W.transpose() * W;
    Eigen::VectorXd b = W.transpose() * W * z_trim;
    auto start = chrono::system_clock::now();
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(A);
    Eigen::VectorXd y_res = cg.solve(b);
    auto end = std::chrono::system_clock::now(); // 計測終了時刻を保存
    auto dur = end - start;                      // 要した時間を計算
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    // 要した時間をミリ秒（1/1000秒）に変換して表示
    std::cout << msec << " milli sec \n";

    auto color_mrf_ptr = make_shared<geometry::PointCloud>();
    vector<vector<double>> interpolated_z(height, vector<double>(width));
    for (int i = 0; i < h_trim; i++)
    {
        for (int j = 0; j < w_trim; j++)
        {
            double z = y_res[i * w_trim + j];
            z = min(z, 100.0);
            z = max(z, 0.0);
            double x = z * (j - width / 2) / f_x;
            double y = z * (i - height / 2) / f_x;
            color_mrf_ptr->points_.emplace_back(Eigen::Vector3d(x, y, z));
            interpolated_z[i][j] = z;
        }
    }

    auto other_mrf_ptr = make_shared<geometry::PointCloud>();
    for (int i = 0; i < h_trim; i++)
    {
        for (int j = 0; j < w_trim; j++)
        {
            double z = params_z[i * width + j];
            double x = z * (j - width / 2) / f_x;
            double y = z * (i - height / 2) / f_x;
            color_mrf_ptr->points_.emplace_back(Eigen::Vector3d(x + 100, y, z));
        }
    }

    { // Evaluation
        double error = 0;
        int cnt = 0;
        int cannot_cnt = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < w_trim; j++)
            {
                if (base_z[i][j] > 0 && params_z[i * width + j] == 0 && interpolated_z[i][j] > 0)
                {
                    error += abs((base_z[i][j] - interpolated_z[i][j]) / base_z[i][j]);
                    cnt++;
                }
                if (base_z[i][j] > 0 && params_z[i * width + j] == 0)
                {
                    cannot_cnt++;
                }
            }
        }
        cout << "W trim = " << w_trim << endl;
        cout << "Error = " << error / cnt << endl;
        cout << "Cannot cnt = " << cannot_cnt - cnt << endl;
    }

    if (see_res)
    {
        Eigen::MatrixXd front(4, 4);
        front << 1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, -1, 0,
            0, 0, 0, 1;
        pcd_ptr->Transform(front);
        filtered_ptr->Transform(front);
        color_mrf_ptr->Transform(front);

        cv::imshow("img", projected_img);
        cv::waitKey();
        visualization::DrawGeometries({color_mrf_ptr}, "PointCloud", 1600, 900);
    }
}

int main(int argc, char *argv[])
{
    //vector<int> data_nos = {550, 1000, 1125, 1260, 1550}; // 03_03_miyanosawa
    //vector<int> data_nos = {10, 20, 30, 40, 50}; // 02_04_13jo
    vector<int> data_nos = {700, 1290, 1460, 2350, 3850}; // 02_04_miyanosawa
    segmentate(700, width, true);

    for (int i = 0; i < data_nos.size(); i++)
    {
        cout << "--- No." << data_nos[i] << " ---" << endl;
        for (int w_trim = 100; w_trim < width; w_trim += 100)
        {
            segmentate(data_nos[i], w_trim, false);
        }
        segmentate(data_nos[i], width, false);
    }

    return 0;
}