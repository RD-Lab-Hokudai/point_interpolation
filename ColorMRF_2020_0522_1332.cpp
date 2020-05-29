#include <iostream>
#include <vector>
#include <stack>
#include <map>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include "eigen3/unsupported/Eigen/NumericalDiff"
#include <time.h>

using namespace std;
using namespace open3d;

const int width = 882;
const int height = 560;
const double f_x = width / 2 * 1.01;

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
    misra1a_functor(int inputs, int values, cv::Vec3b *x, double *y, double k, double c)
        : inputs_(inputs), values_(values), x(x), z(y), k(k), c(c) {}

    cv::Vec3b *x;
    double *z;

    double k;
    double c;

    int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    int operator()(const Eigen::VectorXd &y, Eigen::VectorXd &fvec) const
    {
        fvec[0] = 0;
        for (int i = 0; i < inputs_; i++)
        {
            if (z[i] > 0)
            {
                fvec[0] += k * (y[i] - z[i]) * (y[i] - z[i]);
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
                    w = 1;
                    fvec[0] += w * (y[i] - y[v * width + u]) * (y[i] - y[v * width + u]);
                }
            }
        }
        cout << "error = " << fvec[0] << endl;
        return 0;
    }

    const int inputs_;
    const int values_;
    int inputs() const { return inputs_; }
    int values() const { return values_; }
};

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
                    params_y[v * width + u] = pcd_ptr->points_[i][2];
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
    double k = 1;
    double c = 0.01;
    misra1a_functor functor(length, 1, &params_x[0], &params_z[0], k, c);
    Eigen::NumericalDiff<misra1a_functor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<misra1a_functor>> lm(numDiff);
    cout << "lm" << endl;
    Eigen::VectorXd fvec(1);
    functor.operator()(params_y, fvec);
    int info = lm.minimize(params_y);
    cout << info << endl;
    functor.operator()(params_y, fvec);

    auto color_mrf_ptr = make_shared<geometry::PointCloud>();
    cout << "interpolate" << endl;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double z = params_y[i * width + j];
            double x = z * (j - width / 2) / f_x;
            double y = z * (i - height / 2) / f_x;
            color_mrf_ptr->points_.emplace_back(Eigen::Vector3d(x, y, z));
        }
    }
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