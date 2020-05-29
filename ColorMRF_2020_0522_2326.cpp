#include <iostream>
#include <vector>
#include <stack>
#include <map>

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
    const string img_name = "../3037.png";
    auto img = cv::imread(img_name);

    const string file_name = "../3037.pcd";

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
                    //img.at<cv::Vec3b>(v, u)[0] = 255;
                    //img.at<cv::Vec3b>(v, u)[1] = 0;
                    //img.at<cv::Vec3b>(v, u)[2] = 0;
                    params_z[v * width + u] = pcd_ptr->points_[i][2];
                }
            }
        }

        pcd_ptr->points_[i][0] += 100;
    }

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            params_x[i * width + j] = img.at<cv::Vec3b>(i, j);
        }
    }

    double k = 3;
    double c = 1;
    int w_trim = 10;
    int h_trim = 10;
    int length_trim = w_trim * h_trim;
    Eigen::SparseMatrix<double> W(length, length);
    vector<Eigen::Triplet<double>> W_triplets;
    for (int i = 0; i < length; i++)
    {
        if (params_z[i] > 0)
        {
            W_triplets.emplace_back(i, i, k);
        }
    }
    W.setFromTriplets(W_triplets.begin(), W_triplets.end());

    Eigen::SparseMatrix<double> S(length, length);
    vector<Eigen::Triplet<double>> S_triplets;
    int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double wSum = 0;
            for (int k = 0; k < 8; k++)
            {
                int u = j + dx[k];
                int v = i + dy[k];
                if (0 <= u && u < width && 0 <= v && v < height)
                {
                    double x_norm2 = (params_x[i * width + j][0] - params_x[v * width + u][0]) / 255;
                    x_norm2 = x_norm2 * x_norm2;
                    double w = -sqrt(exp(-c * x_norm2));
                    S_triplets.emplace_back(i * width + j, v * width + u, w);
                    wSum += w;
                }
            }
            S_triplets.emplace_back(i * width + j, i * width + j, -wSum);
        }
    }
    S.setFromTriplets(S_triplets.begin(), S_triplets.end());
    Eigen::SparseMatrix<double> A = S.transpose() * S + W.transpose() * W;
    Eigen::VectorXd b = W.transpose() * W * params_z;
    cout << "calculated" << endl;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(A);
    cout << cg.info() << endl;
    Eigen::VectorXd y_res = cg.solve(b);
    cout << cg.info() << endl;
    cout << cg.iterations() << " " << cg.error() << endl;
    /*
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success)
    {
        cout << solver.info() << endl;
        return 0;
    }
    auto y_res = solver.solve(b);
    if (solver.info() != Eigen::Success)
    {
        cout << solver.info() << endl;
        return 0;
    }
    */

    auto color_mrf_ptr = make_shared<geometry::PointCloud>();
    cout << "interpolate" << endl;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double z = y_res[i * width + j];
            double x = z * (j - width / 2) / f_x;
            double y = z * (i - height / 2) / f_x;
            color_mrf_ptr->points_.emplace_back(Eigen::Vector3d(x, y, z));
        }
    }
    cout << color_mrf_ptr->points_.size() << endl;
    cout << "done" << endl;

    Eigen::MatrixXd front(4, 4);
    front << 1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1;
    pcd_ptr->Transform(front);
    filtered_ptr->Transform(front);
    color_mrf_ptr->Transform(front);

    cv::imshow("a", img);
    cv::waitKey();

    visualization::DrawGeometries({color_mrf_ptr}, "PointCloud", 1600, 900);

    return 0;
}