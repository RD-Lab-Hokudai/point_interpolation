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
double X = 500;
double Y = 474;
double Z = 458;
double theta = 506;
double phi = 527;

void segmentate(int data_no, bool see_res = false)
{
    const string img_name = "../../../data/2020_03_03_miyanosawa_img_pcd/" + to_string(data_no) + ".png";
    const string file_name = "../../../data/2020_03_03_miyanosawa_img_pcd/" + to_string(data_no) + ".pcd";
    const bool vertical = true;

    auto img = cv::imread(img_name);

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

    geometry::PointCloud pointcloud;
    auto pcd_ptr = make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(file_name, pointcloud))
    {
        cout << "Cannot read" << endl;
    }

    *pcd_ptr = pointcloud;
    auto filtered_ptr = make_shared<geometry::PointCloud>();
    vector<vector<double>> base_z(height, vector<double>(width));
    vector<vector<double>> filtered_z(height, vector<double>(width));
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
        if (z > 0)
        {
            int u = (int)(width / 2 + f_x * x / z);
            int v = (int)(height / 2 + f_x * y / z);
            if (0 <= u && u < width && 0 <= v && v < height)
            {
                auto it = lower_bound(tans.begin(), tans.end(), rawY / r);
                int index = it - tans.begin();
                if (index % 4 == 0)
                {
                    filtered_ptr->points_.emplace_back(x, y, z);
                    filtered_z[v][u] = z;
                }
                base_z[v][u] = z;
            }
        }

        pcd_ptr->points_[i][0] += 100;
    }

    auto start = chrono::system_clock::now();
    vector<vector<double>> interpolated_z(height, vector<double>(width));
    if (vertical)
    {
        for (int j = 0; j < width; j++)
        {
            vector<int> up(height, -1);
            for (int i = 0; i < height; i++)
            {
                if (filtered_z[i][j] > 0)
                {
                    up[i] = i;
                }
                else if (i > 0)
                {
                    up[i] = up[i - 1];
                }
            }

            vector<int> down(height, -1);
            for (int i = height - 1; i >= 0; i--)
            {
                if (filtered_z[i][j] > 0)
                {
                    down[i] = i;
                }
                else if (i + 1 < height)
                {
                    down[i] = down[i + 1];
                }
            }

            for (int i = 0; i < height; i++)
            {
                if (up[i] == -1 && down[i] == -1)
                {
                    interpolated_z[i][j] = -1;
                }
                else if (up[i] == -1 || down[i] == -1 || up[i] == i)
                {
                    interpolated_z[i][j] = filtered_z[max(up[i], down[i])][j];
                }
                else
                {
                    interpolated_z[i][j] = (filtered_z[down[i]][j] * (i - up[i]) + filtered_z[up[i]][j] * (down[i] - i)) / (down[i] - up[i]);
                }
            }
        }
        for (int i = 0; i < height; i++)
        {
            vector<int> left(width, -1);
            for (int j = 0; j < width; j++)
            {
                if (interpolated_z[i][j] > 0)
                {
                    left[j] = j;
                }
                else if (j > 0)
                {
                    left[j] = left[j - 1];
                }
            }

            vector<int> right(width, -1);
            for (int j = width - 1; j >= 0; j--)
            {
                if (interpolated_z[i][j] > 0)
                {
                    right[j] = j;
                }
                else if (j + 1 < width)
                {
                    right[j] = right[j + 1];
                }
            }

            for (int j = 0; j < width; j++)
            {
                if (left[j] == -1 && right[j] == -1)
                {
                    interpolated_z[i][j] = -1;
                }
                else if (left[j] == -1 || right[j] == -1 || left[j] == j)
                {
                    interpolated_z[i][j] = interpolated_z[i][max(left[j], right[j])];
                }
                else
                {
                    interpolated_z[i][j] = (interpolated_z[i][right[j]] * (j - left[j]) + interpolated_z[i][left[j]] * (right[j] - j)) / (right[j] - left[j]);
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < height; i++)
        {
            vector<int> left(width, -1);
            for (int j = 0; j < width; j++)
            {
                if (filtered_z[i][j] > 0)
                {
                    left[j] = j;
                }
                else if (j > 0)
                {
                    left[j] = left[j - 1];
                }
            }

            vector<int> right(width, -1);
            for (int j = width - 1; j >= 0; j--)
            {
                if (filtered_z[i][j] > 0)
                {
                    right[j] = j;
                }
                else if (j + 1 < width)
                {
                    right[j] = right[j + 1];
                }
            }

            for (int j = 0; j < width; j++)
            {
                if (left[j] == -1 && right[j] == -1)
                {
                    interpolated_z[i][j] = -1;
                }
                else if (left[j] == -1 || right[j] == -1 || left[j] == j)
                {
                    interpolated_z[i][j] = filtered_z[i][max(left[j], right[j])];
                }
                else
                {
                    interpolated_z[i][j] = (filtered_z[i][right[j]] * (j - left[j]) + filtered_z[i][left[j]] * (right[j] - j)) / (right[j] - left[j]);
                }
            }
        }
        for (int j = 0; j < width; j++)
        {
            vector<int> up(height, -1);
            for (int i = 0; i < height; i++)
            {
                if (interpolated_z[i][j] > 0)
                {
                    up[i] = i;
                }
                else if (i > 0)
                {
                    up[i] = up[i - 1];
                }
            }

            vector<int> down(height, -1);
            for (int i = height - 1; i >= 0; i--)
            {
                if (interpolated_z[i][j] > 0)
                {
                    down[i] = i;
                }
                else if (i + 1 < width)
                {
                    down[i] = down[i + 1];
                }
            }

            for (int i = 0; i < height; i++)
            {
                if (up[i] == -1 && down[i] == -1)
                {
                    interpolated_z[i][j] = -1;
                }
                else if (up[i] == -1 || down[i] == -1 || up[i] == i)
                {
                    interpolated_z[i][j] = interpolated_z[max(up[i], down[i])][j];
                }
                else
                {
                    interpolated_z[i][j] = (interpolated_z[down[i]][j] * (i - up[i]) + interpolated_z[up[i]][j] * (down[i] - i)) / (down[i] - up[i]);
                }
            }
        }
    }
    auto end = chrono::system_clock::now(); // 計測終了時刻を保存
    auto dur = end - start;                 // 要した時間を計算
    auto msec = chrono::duration_cast<chrono::milliseconds>(dur).count();
    // 要した時間をミリ秒（1/1000秒）に変換して表示
    std::cout << msec << " milli sec \n";

    auto linear_interpolation_ptr = make_shared<geometry::PointCloud>();
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (base_z[i][j] == 0)
            {
                continue;
            }

            double z = interpolated_z[i][j];
            if (z == -1)
            {
                continue;
            }
            double x = z * (j - width / 2) / f_x;
            double y = z * (i - height / 2) / f_x;
            linear_interpolation_ptr->points_.emplace_back(x, y, z);
        }
    }

    { // Evaluation
        double error = 0;
        int cnt = 0;
        int cannot_cnt = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (base_z[i][j] > 0 && filtered_z[i][j] == 0 && interpolated_z[i][j] > 0)
                {
                    error += abs((base_z[i][j] - interpolated_z[i][j]) / base_z[i][j]);
                    cnt++;
                }
                if (base_z[i][j] > 0 && filtered_z[i][j] == 0)
                {
                    cannot_cnt++;
                }
            }
        }
        cout << "cannot cnt = " << cannot_cnt - cnt << endl;
        cout << "Error = " << error / cnt << endl;
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
        linear_interpolation_ptr->Transform(front);

        cv::imshow("img", img);
        cv::waitKey();

        visualization::DrawGeometries({linear_interpolation_ptr}, "PointCloud", 1600, 900);
    }
}

int main(int argc, char *argv[])
{
    vector<int> data_nos = {550, 1000, 1125, 1260, 1550};
    for (int i = 0; i < data_nos.size(); i++)
    {
        segmentate(data_nos[i], true);
    }
    return 0;
}