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

// Generic functor
template <typename _Scalar, int NX = Dynamic, int NY = Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum
    {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;
};

struct misra1a_functor : Functor<double>
{
    misra1a_functor(int inputs, int values, cv::Vec3b *x, double *y, double k, double c)
        : inputs_(inputs), values_(values), x(x), z(y), k(k), c(c) {}

    cv::Vec3b *x;
    double *z;

    double k;
    double c;

    int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    int operator()(const VectorXd &y, VectorXd &fvec) const
    {
        for (int i = 0; i < values_; ++i)
        {
            fvec[i] = 0;
            if (z[i] > 0)
            {
                fvec[i] += k * (y[i] - z[i]) * (y[i] - z[i]);
            }

            for (int j = 0; j < 8; j++)
            {
                int u = i % width + dx[j];
                int v = i / width + dy[j];
                if (0 <= u && u < width && 0 <= v && v < height)
                {
                    double x_norm2 = x[i][0] - x[v * width + u][0];
                    x_norm2 = x_norm2 * x_norm2;
                    double w = exp(-c * x_norm2);
                    fvec[i] += w * (y[i] - y[v * width + u]) * (y[i] - y[v * width + u]);
                }
            }
        }
        return 0;
    }

    const int inputs_;
    const int values_;
    int inputs() const { return inputs_; }
    int values() const { return values_; }
};

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

    vector<cv::Vec3b> params_x(width * height);
    Eigen::VectorXd params_y(width * height);
    vector<double> params_z(width * height);

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
                    img.at<cv::Vec3b>(v, u)[0] = 255;
                    img.at<cv::Vec3b>(v, u)[1] = 0;
                    img.at<cv::Vec3b>(v, u)[2] = 0;
                    params_z[v * width + u] = pcd_ptr->points_[i][2];
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

    int length = width * height;
    Eigen::SparseMatrix<double, Eigen::RowMajor, int64_t> A(length, length);
    Eigen::VectorXd b(length);
    int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    vector<Eigen::Triplet<double>> A_triplets;
    double k = 0.001;
    double c = 1;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double yTmp = 0;
            /*
            if (params_z[i * width + j] > 0)
            {
                yTmp += -k;
                b[i * width + j] = -k * params_z[i * width + j];
            }
            else
            {
                b[i * width + j] = 0;
            }
            */
            b[i * width + j] = 0;
            if (params_z[i * width + j] > 0)
            {
                yTmp += k;
                b[i * width + j] = k * params_z[i * width + j];
            }

            for (int k = 0; k < 8; k++)
            {
                int u = j + dx[k];
                int v = i + dy[k];
                if (0 <= u && u < width && 0 <= v && v < height)
                {
                    double x_norm2 = params_x[i * width + j][0] - params_x[v * width + u][0];
                    x_norm2 = x_norm2 * x_norm2;
                    double w = exp(-c * x_norm2);
                    yTmp += w;
                    A_triplets.emplace_back(Eigen::Triplet<double>(i * width + j, v * width + u, -w));
                }
            }
            A_triplets.emplace_back(Eigen::Triplet<double>(i * width + j, i * width + j, yTmp));
        }
    }
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());
    /*
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(A);
    Eigen::VectorXd y_res;
    y_res = cg.solve(b);
    if (cg.info() == Eigen::Success)
    {
        cout << "OK!" << endl;
    }
    cout << "#iterations:     " << cg.iterations() << endl;
    cout << "estimated error: " << cg.error() << endl;

    A.makeCompressed();
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> lu;
    lu.analyzePattern(A);
    cout << "solve" << endl;
    Eigen::VectorXd y_res;
    lu.factorize(A);
    cout << "factorized" << endl;
    if (lu.info() != Eigen::Success)
    {
        cout << "Bad!" << endl;
    }
    y_res = lu.solve(b);
    cout << "what" << endl;
    if (lu.info() == Eigen::Success)
    {
        cout << "OK!" << endl;
    }
    else
    {
        cout << "OUT!" << endl;
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