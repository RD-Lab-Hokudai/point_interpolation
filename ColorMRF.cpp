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

const int width = 882;
const int height = 560;
const double f_x = width / 2 * 1.01;

int main(int argc, char *argv[])
{
    const string img_name = "../81.png";
    auto img = cv::imread(img_name);

    const string file_name = "../81.pcd";

    vector<double> tans;
    double PI = acos(-1);
    double rad = (-16.6 - 0.265) * PI / 180;
    double delta_rad = 0.53 * PI / 180;
    double max_rad = (16.6 + 0.265) * PI / 180;
    while (rad < max_rad - 0.000001)
    {
        rad += delta_rad;
        tans.emplace_back(tan(rad));
    }

    int length = width * height;
    vector<cv::Vec3b> params_x(length);
    Eigen::VectorXd params_z(length);
    vector<vector<double>> base_z(height, vector<double>(width));

    geometry::PointCloud pointcloud;
    auto pcd_ptr = make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(file_name, pointcloud))
    {
        cout << "Cannot read" << endl;
    }

    *pcd_ptr = pointcloud;
    auto filtered_ptr = make_shared<geometry::PointCloud>();
    for (int i = 0; i < pcd_ptr->points_.size(); i++)
    {
        double x = pcd_ptr->points_[i][1];
        double y = -pcd_ptr->points_[i][2];
        double z = -pcd_ptr->points_[i][0];
        pcd_ptr->points_[i] = Eigen::Vector3d(x, y, z);
        if (pcd_ptr->points_[i][2] > 0)
        {
            int u = (int)(width / 2 + f_x * x / z);
            int v = (int)(height / 2 + f_x * y / z);
            if (0 <= u && u < width && 0 <= v && v < height)
            {
                auto it = lower_bound(tans.begin(), tans.end(), y / z);
                int index = it - tans.begin();
                if (index % 4 == 0)
                {
                    filtered_ptr->points_.emplace_back(pcd_ptr->points_[i]);
                    params_z[v * width + u] = pcd_ptr->points_[i][2];
                }
                base_z[v][u] = pcd_ptr->points_[i][2];
            }
        }

        pcd_ptr->points_[i][0] += 5;
    }

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            params_x[i * width + j] = img.at<cv::Vec3b>(i, j);
        }
    }

    double k = 1;
    double c = 1000;
    int w_trim = 10;
    int h_trim = height;
    int length_trim = w_trim * h_trim;
    Eigen::VectorXd z_trim(length_trim);
    Eigen::VectorXd y_res(length_trim);
    Eigen::SparseMatrix<double> W(length_trim, length_trim);
    vector<Eigen::Triplet<double>> W_triplets;
    double z_sum = 0;
    int z_cnt = 0;
    for (int i = 0; i < h_trim; i++)
    {
        for (int j = 0; j < w_trim; j++)
        {
            if (params_z[i * width + j] > 0)
            {
                z_trim[i * w_trim + j] = params_z[i * width + j];
                W_triplets.emplace_back(i * w_trim + j, i * w_trim + j, k);
                y_res[i * w_trim + j] = params_z[i * width + j];
                z_sum += params_z[i * width + j];
                z_cnt++;
            }
        }
    }
    /*
    for (int i = 0; i < length_trim; i++)
    {
        if (y_res[i] == 0)
        {
            y_res[i] = z_sum / z_cnt;
        }
    }
    */
    W.setFromTriplets(W_triplets.begin(), W_triplets.end());

    Eigen::SparseMatrix<double> S(length_trim, length_trim);
    vector<Eigen::Triplet<double>> S_triplets;
    int dires = 4;
    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};
    //int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    //int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
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
    cout << "A" << endl;
    Eigen::VectorXd b = W.transpose() * W * z_trim;
    cout << "calculated" << endl;
    auto start = chrono::system_clock::now();
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(A);
    cout << cg.info() << endl;
    y_res = cg.solveWithGuess(b, y_res);
    cout << cg.info() << endl;
    cout << cg.iterations() << " " << cg.error() << endl;

    auto color_mrf_ptr = make_shared<geometry::PointCloud>();
    cout << "interpolate" << endl;
    double interpolation_error = 0;
    int base_cnt = 0;
    for (int i = 0; i < h_trim; i++)
    {
        for (int j = 0; j < w_trim; j++)
        {
            double z = y_res[i * w_trim + j];
            double x = z * (j - width / 2) / f_x;
            double y = z * (i - height / 2) / f_x;
            color_mrf_ptr->points_.emplace_back(Eigen::Vector3d(x, y, z));
            if (base_z[i][j] > 0 && z > 0)
            {
                interpolation_error += abs((base_z[i][j] - z) / base_z[i][j]);
                base_cnt++;
            }
        }
    }
    cout << "Error = " << interpolation_error / base_cnt << endl;

    auto end = chrono::system_clock::now(); // 計測終了時刻を保存
    auto msec = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    // 要した時間をミリ秒（1/1000秒）に変換して表示
    cout << msec << " milli sec \n";

    Eigen::MatrixXd front(4, 4);
    front << 1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1;
    pcd_ptr->Transform(front);
    filtered_ptr->Transform(front);
    color_mrf_ptr->Transform(front);

    //cv::imshow("a", img);
    //cv::waitKey();

    visualization::DrawGeometries({pcd_ptr, color_mrf_ptr}, "PointCloud", 1600, 900);

    return 0;
}