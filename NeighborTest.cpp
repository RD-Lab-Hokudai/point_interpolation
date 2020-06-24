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
double X = 500;
double Y = 474;
double Z = 458;
double theta = 506;
double phi = 527;

void calcNormals(shared_ptr<geometry::PointCloud> pcd_ptr)
{
}

void calcNeighbors(shared_ptr<geometry::PointCloud> pcd_ptr)
{
}

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
    vector<vector<Eigen::Vector3d>> layers(16, vector<Eigen::Vector3d>());
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
        /*
        double x = rawX;
        double y = rawY;
        double z = rawZ;
        */
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
                    layers[index / 4].emplace_back(x, y, z);
                }
                base_z[v][u] = z;
            }
        }

        pcd_ptr->points_[i][0] += 100;
    }

    vector<shared_ptr<geometry::PointCloud>> layer_pcds(16);
    Eigen::MatrixXd front(4, 4);
    front << 1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1;
    for (int i = 0; i < 16; i++)
    {
        sort(begin(layers[i]), end(layers[i]),
             [](Eigen::Vector3d a, Eigen::Vector3d b) { return a[0] / a[2] < b[0] / b[2]; });
        layer_pcds[i] = make_shared<geometry::PointCloud>();
        int prevU, prevV;
        for (int j = 0; j < layers[i].size(); j++)
        {
            layer_pcds[i]->points_.emplace_back(layers[i][j]);
            int c_val = j * 10;
            int u = (int)(width / 2 + f_x * layers[i][j][0] / layers[i][j][2]);
            int v = (int)(height / 2 + f_x * layers[i][j][1] / layers[i][j][2]);
            if (j > 0)
            {
                //cv::line(img, cv::Point(prevU, prevV), cv::Point(u, v), cv::Scalar(255, 0, 0), 2, 4);
            }
            prevU = u;
            prevV = v;
            img.at<cv::Vec3b>(v, u)[0] = 255;
            layer_pcds[i]->colors_.emplace_back((c_val % 256) / 255.0, (c_val / 256 % 256) / 255.0, (c_val / 256 / 256 % 256) / 255.0);
        }
        layer_pcds[i]->Transform(front);
    }

    vector<vector<double>> interpolated_z(height, vector<double>(width));
    for (int i = 0; i < 16; i++)
    {
        int prevU, prevV;
        for (int j = 0; j < layers[i].size(); j++)
        {
            int c_val = j * 10;
            int u = (int)(width / 2 + f_x * layers[i][j][0] / layers[i][j][2]);
            int v = (int)(height / 2 + f_x * layers[i][j][1] / layers[i][j][2]);
            interpolated_z[v][u] = layers[i][j][2];
            if (j > 0)
            {
                float delta = 1.0f * (v - prevV) / (u - prevU);
                float vF = prevV + delta;
                int uTmp = prevU + 1;
                while (uTmp < u)
                {
                    //img.at<cv::Vec3b>((int)vF, uTmp)[0] = 255;
                    interpolated_z[(int)vF][uTmp] = (interpolated_z[v][u] * (uTmp - prevU) + interpolated_z[prevV][prevU] * (u - uTmp)) / (u - prevU);
                    uTmp++;
                    vF += delta;
                }
            }
            prevU = u;
            prevV = v;
        }
    }

    vector<vector<vector<int>>> neighbors(16);
    {
        auto start = chrono::system_clock::now();
        int point_cnt = 0;
        for (int i = 0; i < 16; i++)
        {
            neighbors[i] = vector<vector<int>>(layers[i].size());
        }
        // Find neighbors
        for (int i = 0; i < 15; i++)
        {
            for (int j = 0; j < layers[i].size(); j++)
            {
                int u = (int)(width / 2 + f_x * layers[i][j][0] / layers[i][j][2]);
                int v = (int)(height / 2 + f_x * layers[i][j][1] / layers[i][j][2]);
                int u0 = (int)(width / 2 + f_x * layers[i + 1][0][0] / layers[i + 1][0][2]);
                if (u0 > u)
                {
                    int v0 = (int)(height / 2 + f_x * layers[i + 1][0][1] / layers[i + 1][0][2]);
                    cv::line(img, cv::Point(u, v), cv::Point(u0, v0), cv::Scalar(0, 255, 0), 1, 4);
                    int toI = point_cnt + layers[i].size();
                    neighbors[i][j].emplace_back(toI);
                    neighbors[i + 1][0].emplace_back(point_cnt + j);
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
                        cv::line(img, cv::Point(u, v), cv::Point(u2, v2), cv::Scalar(0, 255, 0), 1, 4);
                        int toI = point_cnt + layers[i].size() + ii;
                        neighbors[i][j].emplace_back(toI);
                        neighbors[i + 1][ii].emplace_back(point_cnt + j);
                    }
                }
                if (j + 1 < layers[i].size())
                {
                    neighbors[i][j].emplace_back(point_cnt + j + 1);
                    neighbors[i][j + 1].emplace_back(point_cnt + j);
                }
                neighbors[i][j].emplace_back(point_cnt + j); // Contains myself
            }
            point_cnt += layers[i].size();
        }

        auto end = chrono::system_clock::now();
        cout << "Neighboring time = " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }

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
                    interpolated_z[i][j] = filtered_z[max(up[i], down[i])][j];
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

    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (base_z[i][j] == 0)
            {
                continue;
            }

            /*if (interpolated_z[i][j] == 0)
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
            interpolated_ptr->points_.emplace_back(x, y, z);
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

    auto sorted_ptr = make_shared<geometry::PointCloud>();
    {
        vector<pair<int, int>> correspondences;
        int point_cnt = 0;
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < layers[i].size(); j++)
            {
                sorted_ptr->points_.emplace_back(layers[i][j]);
                int from = point_cnt + j;
                for (int k = 0; k < neighbors[i][j].size(); k++)
                {
                    if (from < neighbors[i][j][k])
                    {
                        correspondences.emplace_back(make_pair(from, neighbors[i][j][k]));
                    }
                }
            }
            point_cnt += layers[i].size();
        }

        auto lineset_ptr = geometry::LineSet::CreateFromPointCloudCorrespondences(
            *sorted_ptr, *sorted_ptr, correspondences);
        cout << "Make" << endl;
        lineset_ptr->Transform(front);
        visualization::DrawGeometries({lineset_ptr}, "LINE", 1600, 900);
    }

    { // Calc normals
        auto start = chrono::system_clock::now();

        int point_cnt = 0;
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < layers[i].size(); j++)
            {
                Eigen::Vector3d pa = Eigen::Vector3d::Zero();
                for (int k = 0; k < neighbors[i][j].size(); k++)
                {
                    pa += sorted_ptr->points_[neighbors[i][j][k]];
                }
                pa /= neighbors[i][j].size();
                Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
                for (int k = 0; k < neighbors[i][j].size(); k++)
                {
                    for (int ii = 0; ii < 3; ii++)
                    {
                        for (int jj = 0; jj < 3; jj++)
                        {
                            Q(ii, jj) += (sorted_ptr->points_[neighbors[i][j][k]][ii] - pa[ii]) * (sorted_ptr->points_[neighbors[i][j][k]][jj] - pa[jj]);
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
            point_cnt += layers[i].size();
        }

        auto end = chrono::system_clock::now();
        cout << "Normaling time = " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }

    interpolated_ptr->Transform(front);
    sorted_ptr->Transform(front);
    visualization::DrawGeometries({sorted_ptr}, "a", 1600, 900);

    cv::imshow("hoge", img);
    cv::waitKey();
    visualization::DrawGeometries({interpolated_ptr /* layer_pcds[0], layer_pcds[1], layer_pcds[2], layer_pcds[3], layer_pcds[4]*/}, "PointCloud", 1600, 900);
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