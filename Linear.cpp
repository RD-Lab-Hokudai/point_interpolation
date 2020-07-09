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

shared_ptr<geometry::PointCloud> calc_filtered(shared_ptr<geometry::PointCloud> raw_pcd_ptr,
                                               vector<vector<double>> &base_z, vector<vector<double>> &filtered_z,
                                               vector<vector<int>> &neighbors, int layer_cnt = 16)
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

    base_z = vector<vector<double>>(height, vector<double>(width));
    filtered_z = vector<vector<double>>(height, vector<double>(width));
    vector<vector<Eigen::Vector3d>> layers(layer_cnt, vector<Eigen::Vector3d>());
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

        if (z > 0)
        {
            int u = (int)(width / 2 + f_x * x / z);
            int v = (int)(height / 2 + f_x * y / z);
            if (0 <= u && u < width && 0 <= v && v < height)
            {
                auto it = lower_bound(tans.begin(), tans.end(), rawY / r);
                int index = it - tans.begin();
                if (index % (64 / layer_cnt) == 0)
                {
                    int id = index / (64 / layer_cnt);
                    layers[index / (64 / layer_cnt)].emplace_back(x, y, z);
                    filtered_z[v][u] = z;
                }
                base_z[v][u] = z;
            }
        }
    }

    int filtered_cnt = 0;
    for (int i = 0; i < layer_cnt; i++)
    {
        /*
        sort(begin(layers[i]), end(layers[i]),
             [](Eigen::Vector3d a, Eigen::Vector3d b) { return a[0] / a[2] < b[0] / b[2]; });
*/
        vector<Eigen::Vector3d> removed;
        for (size_t j = 0; j < layers[i].size(); j++)
        {
            while (removed.size() > 0 && removed.back()[0] / removed.back()[2] > layers[i][j][0] / layers[i][j][2])
            {
                removed.pop_back();
            }
            removed.emplace_back(layers[i][j]);
        }
        layers[i] = removed;
        filtered_cnt += layers[i].size();
    }

    neighbors = vector<vector<int>>(filtered_cnt, vector<int>());
    cv::Mat layer_img = cv::Mat::zeros(height, width, CV_8UC3);
    {
        int point_cnt = 0;
        // Find neighbors
        for (int i = 0; i + 1 < layer_cnt; i++)
        {
            int uPrev = 0;
            int vPrev = 0;
            for (int j = 0; j < layers[i].size(); j++)
            {
                int u = (int)(width / 2 + f_x * layers[i][j][0] / layers[i][j][2]);
                int v = (int)(height / 2 + f_x * layers[i][j][1] / layers[i][j][2]);

                layer_img.at<cv::Vec3b>(v, u) = cv::Vec3b(255 * (i % 2), 255 * ((i + 1) % 2), 0);

                if (j > 0)
                {
                    cv::line(layer_img, cv::Point(uPrev, vPrev), cv::Point(u, v), cv::Scalar(255, 0, 0), 1, 8);
                }
                uPrev = u;
                vPrev = v;

                int u0 = (int)(width / 2 + f_x * layers[i + 1][0][0] / layers[i + 1][0][2]);
                if (u0 > u)
                {
                    int v0 = (int)(height / 2 + f_x * layers[i + 1][0][1] / layers[i + 1][0][2]);
                    int from = point_cnt + j;
                    int to = point_cnt + layers[i].size();

                    neighbors[from].emplace_back(to);
                    neighbors[to].emplace_back(from);
                }
                else
                {
                    int bottom = 0;
                    int top = layers[i + 1].size();
                    while (bottom + 1 < top)
                    {
                        int mid = (bottom + top) / 2;
                        int uTmp = (int)(width / 2 + f_x * layers[i + 1][mid][0] / layers[i + 1][mid][2]);

                        if (uTmp <= u)
                        {
                            bottom = mid;
                        }
                        else
                        {
                            top = mid;
                        }
                    }
                    for (int ii = max(bottom - 1, 0); ii < min(bottom + 2, (int)layers[i + 1].size()); ii++)
                    {
                        int u2 = (int)(width / 2 + f_x * layers[i + 1][ii][0] / layers[i + 1][ii][2]);
                        int v2 = (int)(height / 2 + f_x * layers[i + 1][ii][1] / layers[i + 1][ii][2]);
                        int from = point_cnt + j;
                        int to = point_cnt + layers[i].size() + ii;
                        neighbors[from].emplace_back(to);
                        neighbors[to].emplace_back(from);
                    }
                }
                if (j + 1 < layers[i].size())
                {
                    neighbors[point_cnt + j].emplace_back(point_cnt + j + 1);
                    neighbors[point_cnt + j + 1].emplace_back(point_cnt + j);
                }
                neighbors[point_cnt + j].emplace_back(point_cnt + j); // Contains myself
            }
            point_cnt += layers[i].size();
        }
    }

    cv::imshow("U", layer_img);
    cv::waitKey();

    auto sorted_ptr = make_shared<geometry::PointCloud>();
    {
        for (int i = 0; i < layer_cnt; i++)
        {
            for (int j = 0; j < layers[i].size(); j++)
            {
                sorted_ptr->points_.emplace_back(layers[i][j]);
            }
        }
    }

    {
        int point_cnt = 0;
        for (int i = 0; i < sorted_ptr->points_.size(); i++)
        {
            Eigen::Vector3d pa = Eigen::Vector3d::Zero();
            for (int j = 0; j < neighbors[i].size(); j++)
            {
                pa += sorted_ptr->points_[neighbors[i][j]];
            }
            pa /= neighbors[i].size();
            Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
            for (int j = 0; j < neighbors[i].size(); j++)
            {
                for (int ii = 0; ii < 3; ii++)
                {
                    for (int jj = 0; jj < 3; jj++)
                    {
                        Q(ii, jj) += (sorted_ptr->points_[neighbors[i][j]][ii] - pa[ii]) * (sorted_ptr->points_[neighbors[i][j]][jj] - pa[jj]);
                    }
                }
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> ES(Q);
            if (ES.info() != Eigen::Success)
            {
                continue;
            }

            sorted_ptr->normals_.emplace_back(ES.eigenvectors().col(0));
        }
    }

    return sorted_ptr;
}

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

    vector<vector<double>> base_z, filtered_z;
    vector<vector<int>> neighbors;
    int layer_cnt = 16;
    shared_ptr<geometry::PointCloud> filtered_ptr = calc_filtered(pcd_ptr, base_z, filtered_z, neighbors, layer_cnt);

    auto start = chrono::system_clock::now();
    vector<vector<double>> interpolated_z(height, vector<double>(width));

    { // Interpolate layer
        cv::Mat layer_img = cv::Mat::zeros(height, width, CV_8UC3);
        for (int i = 0; i < layer_cnt; i++)
        {
            for (int j = 0; j + 1 < filtered_ptr->points_.size(); j++)
            {
                int u = (int)(width / 2 + f_x * filtered_ptr->points_[j][0] / filtered_ptr->points_[j][2]);
                int v = (int)(height / 2 + f_x * filtered_ptr->points_[j][1] / filtered_ptr->points_[j][2]);
                int toU = (int)(width / 2 + f_x * filtered_ptr->points_[j + 1][0] / filtered_ptr->points_[j + 1][2]);
                int toV = (int)(height / 2 + f_x * filtered_ptr->points_[j + 1][1] / filtered_ptr->points_[j + 1][2]);

                if (toU < u)
                {
                    continue;
                }

                float delta = 1.0f * (toV - v) / (toU - u);
                int tmpU = u;
                float tmpV = v;
                while (tmpU <= toU)
                {
                    interpolated_z[(int)tmpV][tmpU] = (filtered_z[toV][toU] * (tmpU - u) + filtered_z[v][u] * (toU - tmpU)) / (toU - u);
                    layer_img.at<cv::Vec3b>((int)tmpV, tmpU)[0] = 255;
                    tmpU++;
                    tmpV += delta;
                }
            }
        }

        //cv::imshow("a", layer_img);
        //cv::waitKey();
    }

    if (vertical)
    {
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
                    interpolated_z[i][j] = interpolated_z[max(up[i], down[i])][j];
                }
                else
                {
                    interpolated_z[i][j] = (interpolated_z[down[i]][j] * (i - up[i]) + interpolated_z[up[i]][j] * (down[i] - i)) / (down[i] - up[i]);
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
            /*
            if (base_z[i][j] == 0)
            {
                continue;
            }*/

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
    cout << linear_interpolation_ptr->points_.size() << endl;

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

        visualization::DrawGeometries({linear_interpolation_ptr}, "PointCloud", 1600, 900);
    }
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