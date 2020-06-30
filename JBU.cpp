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
int X = 498;
int Y = 485;
int Z = 509;
int theta = 483;
int phi = 518;
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
        sort(begin(layers[i]), end(layers[i]),
             [](Eigen::Vector3d a, Eigen::Vector3d b) { return a[0] / a[2] < b[0] / b[2]; });
        filtered_cnt += layers[i].size();
    }

    neighbors = vector<vector<int>>(filtered_cnt, vector<int>());
    {
        int point_cnt = 0;
        // Find neighbors
        for (int i = 0; i + 1 < layer_cnt; i++)
        {
            for (int j = 0; j < layers[i].size(); j++)
            {
                int u = (int)(width / 2 + f_x * layers[i][j][0] / layers[i][j][2]);
                int v = (int)(height / 2 + f_x * layers[i][j][1] / layers[i][j][2]);
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
    const string pcd_path = "../../../data/2020_02_04_miyanosawa/" + to_string(data_no) + ".pcd";
    const string img_path = "../../../data/2020_02_04_miyanosawa/" + to_string(data_no) + ".png";

    cv::Mat img = cv::imread(img_path);

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
    cv::Mat range_img = cv::Mat::zeros(height, width, CV_8UC1);
    {
        vector<vector<int>> up(height, vector<int>(width, -1));
        vector<vector<int>> down(height, vector<int>(width, -1));
        vector<vector<int>> left(height, vector<int>(width, -1));
        vector<vector<int>> right(height, vector<int>(width, -1));
        vector<vector<int>> up_left(height, vector<int>(width, -1));
        vector<vector<int>> up_right(height, vector<int>(width, -1));
        vector<vector<int>> down_left(height, vector<int>(width, -1));
        vector<vector<int>> down_right(height, vector<int>(width, -1));
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (filtered_z[i][j] > 0)
                {
                    int id = i * width + j;
                    up[i][j] = id;
                    down[i][j] = id;
                    left[i][j] = id;
                    right[i][j] = id;
                    up_left[i][j] = id;
                    up_right[i][j] = id;
                    down_left[i][j] = id;
                    down_right[i][j] = id;
                }
                else
                {
                    if (i > 0)
                    {
                        up[i][j] = up[i - 1][j];
                    }
                    if (i + 1 < height)
                    {
                        down[i][j] = down[i + 1][j];
                    }
                    if (j > 0)
                    {
                        left[i][j] = left[i][j - 1];
                    }
                    if (j + 1 < width)
                    {
                        right[i][j] = right[i][j + 1];
                    }
                    if (i > 0 && j > 0)
                    {
                        up_left[i][j] = up_left[i - 1][j - 1];
                    }
                    if (i > 0 && j + 1 < width)
                    {
                        up_right[i][j] = up_right[i - 1][j + 1];
                    }
                    if (i + 1 < height && j > 0)
                    {
                        down_left[i][j] = down_left[i + 1][j - 1];
                    }
                    if (i + 1 < height && j + 1 < width)
                    {
                        down_right[i][j] = down_right[i + 1][j + 1];
                    }
                }
            }
        }

        double min_depth = 10000;
        double max_depth = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (filtered_z[i][j] > 0)
                {
                    min_depth = min(min_depth, filtered_z[i][j]);
                    max_depth = max(max_depth, filtered_z[i][j]);
                }
            }
        }

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int idx = -1;
                if (up[i][j] > 0 && (idx == -1 || abs(up[i][j] / width - i) < abs(idx / width - i)))
                {
                    idx = up[i][j];
                }
                if (down[i][j] > 0 && (idx == -1 || abs(down[i][j] / width - i) < abs(idx / width - i)))
                {
                    idx = down[i][j];
                }
                if (left[i][j] > 0 && (idx == -1 || abs(left[i][j] % width - j) < abs(idx % width - j)))
                {
                    idx = left[i][j];
                }
                if (right[i][j] > 0 && (idx == -1 || abs(right[i][j] % width - j) < abs(idx % width - j)))
                {
                    idx = right[i][j];
                }
                if (up_left[i][j] > 0 && (idx == -1 || abs(max(up_left[i][j] / width, up_left[i][j] % width) - j) < abs(max(idx / width, idx % width) - j)))
                {
                    idx = up_left[i][j];
                }
                if (up_right[i][j] > 0 && (idx == -1 || abs(max(up_right[i][j] / width, up_right[i][j] % width) - j) < abs(max(idx / width, idx % width) - j)))
                {
                    idx = up_right[i][j];
                }
                if (down_left[i][j] > 0 && (idx == -1 || abs(max(down_left[i][j] / width, down_left[i][j] % width) - j) < abs(max(idx / width, idx % width) - j)))
                {
                    idx = down_left[i][j];
                }
                if (down_right[i][j] > 0 && (idx == -1 || abs(max(down_right[i][j] / width, down_right[i][j] % width) - j) < abs(max(idx / width, idx % width) - j)))
                {
                    idx = down_right[i][j];
                }

                if (idx >= 0)
                {
                    range_img.at<unsigned char>(i, j) = (unsigned char)(255 * (filtered_z[idx / width][idx % width] - min_depth) / (max_depth - min_depth));
                }
            }
        }

        cv::imshow("a", range_img);
        cv::waitKey();
    }

    /*
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
    */
}

int main(int argc, char *argv[])
{
    //vector<int> data_nos = {550, 1000, 1125, 1260, 1550};
    //vector<int> data_nos = {10, 20, 30, 40, 50};
    vector<int> data_nos = {700, 1290, 1460, 2350, 3850};
    for (int i = 0; i < data_nos.size(); i++)
    {
        segmentate(data_nos[i], true);
    }
    return 0;
}