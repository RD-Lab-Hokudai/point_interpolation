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
const double f_x = width / 2 * 1.01;

struct point
{
    double x;
    double y;
    double z;
    int u;
    int v;
    bool locked;

    point(double x, double y, double z)
        : x(x), y(y), z(z)
    {
        u = -1;
        v = -1;
        locked = false;
    }
};

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
    misra1a_functor(int values, double *x, double *y, double *z, int *u, int *v)
        : inputs_(5), values_(values), x(x), y(y), z(z), u(u), v(v) {}

    double *x;
    double *y;
    double *z;
    int *u;
    int *v;
    int operator()(const Eigen::VectorXd &params, Eigen::VectorXd &fvec) const
    {
        for (int i = 0; i < values_; ++i)
        {
            double r = sqrt(x[i] * x[i] + z[i] * z[i]);
            double thetaVal = (params[3] - 500) / 1000.0;
            double phiVal = (params[4] - 500) / 1000.0;
            double xp = (x[i] * cos(phiVal) - y[i] * sin(phiVal)) * cos(thetaVal) - (z[i] * cos(phiVal) - y[i] * sin(phiVal)) * sin(thetaVal);
            double yp = y[i] * cos(phiVal) + r * sin(phiVal);
            double zp = (x[i] * cos(phiVal) - y[i] * sin(phiVal)) * sin(thetaVal) + (z[i] * cos(phiVal) - y[i] * sin(phiVal)) * cos(thetaVal);
            double xVal = xp + (params[0] - 500) / 100.0;
            double yVal = yp + (params[1] - 500) / 100.0;
            double zVal = zp + (params[2] - 500) / 100.0;
            int uVal = (int)(width / 2 + f_x * xVal / zVal);
            int vVal = (int)(height / 2 + f_x * yVal / zVal);
            fvec[i] = sqrt((u[i] - uVal) * (u[i] - uVal) + (v[i] - vVal) * (v[i] - vVal));
        }
        return 0;
    }
    const int inputs_;
    const int values_;
    int inputs() const { return inputs_; }
    int values() const { return values_; }
};

vector<cv::Mat> imgs;
vector<shared_ptr<open3d::geometry::PointCloud>> pcd_ptrs;
vector<vector<point>> point_maps;
cv::Mat reprojected;
cv::Mat id_img;

int dataNo = 0;
int X = 498;
int Y = 485;
int Z = 509;
int theta = 483;
int phi = 518;
int calibrateState = 0;
int u0 = 0;
int v0 = 0;
int rate = 1;
int selectedID = -1;

void reproject()
{
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
                    }
                }
            }
            id_img.at<unsigned short>(i, j) = 0;
        }
    }
    for (int i = 0; i < point_maps[dataNo].size(); i++)
    {
        if (point_maps[dataNo][i].locked)
        {
            int u = point_maps[dataNo][i].u;
            int v = point_maps[dataNo][i].v;
            id_img.at<unsigned short>(v, u) = i + 1;
            for (int k1 = 0; k1 < rate; k1++)
            {
                for (int k2 = 0; k2 < rate; k2++)
                {
                    reprojected.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[0] = 255;
                    reprojected.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[1] = 255;
                    reprojected.at<cv::Vec3b>((v - v0) * rate + k1, (u - u0) * rate + k2)[2] = 255;
                }
            }
            continue;
        }

        double rawX = point_maps[dataNo][i].x;
        double rawY = point_maps[dataNo][i].y;
        double rawZ = point_maps[dataNo][i].z;

        double r = sqrt(rawX * rawX + rawZ * rawZ);
        double thetaVal = (theta - 500) / 1000.0;
        double phiVal = (phi - 500) / 1000.0;
        double xp = (rawX * cos(phiVal) - rawY * sin(phiVal)) * cos(thetaVal) - (rawZ * cos(phiVal) - rawY * sin(phiVal)) * sin(thetaVal);
        double yp = rawY * cos(phiVal) + r * sin(phiVal);
        double zp = (rawX * cos(phiVal) - rawY * sin(phiVal)) * sin(thetaVal) + (rawZ * cos(phiVal) - rawY * sin(phiVal)) * cos(thetaVal);
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
                point_maps[dataNo][i].u = u;
                point_maps[dataNo][i].v = v;
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
void on_trackbarTheta(int val, void *object)
{
    theta = val;
    reproject();
}
void on_trackbarPhi(int val, void *object)
{
    phi = val;
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
void on_trackbarCalibrate(int val, void *object)
{
    Eigen::VectorXd params(5);
    params << X, Y, Z, theta, phi;
    vector<double> xs, ys, zs;
    vector<int> us, vs;
    for (int i = 0; i < point_maps[dataNo].size(); i++)
    {
        if (point_maps[dataNo][i].locked)
        {
            xs.emplace_back(point_maps[dataNo][i].x);
            ys.emplace_back(point_maps[dataNo][i].y);
            zs.emplace_back(point_maps[dataNo][i].z);
            us.emplace_back(point_maps[dataNo][i].u);
            vs.emplace_back(point_maps[dataNo][i].v);
        }
    }
    if (xs.size() < 5)
    {
        return;
    }

    misra1a_functor functor(xs.size(), &xs[0], &ys[0], &zs[0], &us[0], &vs[0]);
    Eigen::NumericalDiff<misra1a_functor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<misra1a_functor>> lm(numDiff);
    lm.minimize(params);
    X = params[0];
    Y = params[1];
    Z = params[2];
    theta = params[3];
    phi = params[4];
    cout << params << endl;
}
void mouse_callback(int event, int x, int y, int flags, void *object)
{
    int v = v0 + y / rate;
    int u = u0 + x / rate;
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        if (id_img.at<unsigned short>(v, u) > 0)
        {
            selectedID = id_img.at<unsigned short>(v, u) - 1;
            cout << selectedID << endl;
            point_maps[dataNo][selectedID].locked = !point_maps[dataNo][selectedID].locked;
            cout << selectedID << " " << point_maps[dataNo][selectedID].locked << endl;
        }
        cout << x << " " << y << " clicked" << endl;
    }
    if (event == CV_EVENT_MOUSEMOVE)
    {
        if (selectedID >= 0)
        {
            point_maps[dataNo][selectedID].u = u;
            point_maps[dataNo][selectedID].v = v;
            reproject();
        }
    }
    if (event == CV_EVENT_LBUTTONUP)
    {
        selectedID = -1;
        cout << "upped" << endl;
    }
}

int main(int argc, char *argv[])
{
    vector<int> data_ids = {10, 20, 30, 40, 50};
    for (int i = 0; i < data_ids.size(); i++)
    {
        string img_path = "../../../data/2020_02_04_13jo/" + to_string(data_ids[i]) + ".png";
        imgs.emplace_back(cv::imread(img_path));

        string pcd_path = "../../../data/2020_02_04_13jo/" + to_string(data_ids[i]) + ".pcd";
        open3d::geometry::PointCloud pointcloud;
        vector<point> points;
        if (!open3d::io::ReadPointCloud(pcd_path, pointcloud))
        {
            cout << "Cannot read" << endl;
        }
        for (int i = 0; i < pointcloud.points_.size(); i++)
        {
            double x = pointcloud.points_[i][1];
            double y = -pointcloud.points_[i][2];
            double z = -pointcloud.points_[i][0];
            points.emplace_back(x, y, z);
        }
        point_maps.emplace_back(points);
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
    cv::createTrackbar("theta(-1,1)", "Image", &theta, 1000, &on_trackbarTheta);
    cv::createTrackbar("phi(-1,1)", "Image", &phi, 1000, &on_trackbarPhi);
    cv::createTrackbar("Calibrate", "Image", &calibrateState, 1, &on_trackbarCalibrate);
    cv::setMouseCallback("Image", mouse_callback);

    id_img = cv::Mat::zeros(height, width, CV_16SC1);
    reprojected = cv::Mat::zeros(height, width, CV_8UC3);
    reproject();

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}