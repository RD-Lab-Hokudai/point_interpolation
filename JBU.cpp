#include <iostream>
#include <vector>
#include <stack>
#include <map>
#include <chrono>
#include <queue>

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
// 02_19_13jo
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

    vector<vector<Eigen::Vector3d>> all_layers(64, vector<Eigen::Vector3d>());
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
                all_layers[index].emplace_back(x, y, z);
            }
        }
    }

    int filtered_cnt = 0;
    base_z = vector<vector<double>>(height, vector<double>(width));
    filtered_z = vector<vector<double>>(height, vector<double>(width));
    vector<vector<Eigen::Vector3d>> layers;
    for (int i = 0; i < 64; i++)
    {
        // no sort
        vector<Eigen::Vector3d> removed;
        for (size_t j = 0; j < all_layers[i].size(); j++)
        {
            while (removed.size() > 0 && removed.back()[0] / removed.back()[2] > all_layers[i][j][0] / all_layers[i][j][2])
            {
                removed.pop_back();
            }
            removed.emplace_back(all_layers[i][j]);
        }

        if (i % (64 / layer_cnt) == 0)
        {
            layers.push_back(removed);
            filtered_cnt += removed.size();
        }

        for (size_t j = 0; j < removed.size(); j++)
        {
            int u = (int)(width / 2 + f_x * removed[j][0] / removed[j][2]);
            int v = (int)(height / 2 + f_x * removed[j][1] / removed[j][2]);
            if (i % (64 / layer_cnt) == 0)
            {
                filtered_z[v][u] = removed[j][2];
            }
            base_z[v][u] = removed[j][2];
        }
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

double segmentate(int data_no, double sigma_c = 1, double sigma_s = 15, double sigma_r = 20, int r = 10, bool see_res = false)
{
    const string pcd_path = "../../../data/2020_02_04_13jo/" + to_string(data_no) + ".pcd";
    const string img_path = "../../../data/2020_02_04_13jo/" + to_string(data_no) + ".png";

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
    cv::Mat range_img = cv::Mat::zeros(height, width, CV_16UC3);
    double min_depth = 10000;
    double max_depth = 0;
    {
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

        queue<int> que;
        vector<vector<int>> costs(height, vector<int>(width, 100000));
        vector<vector<int>> cnts(height, vector<int>(width, 0));
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (filtered_z[i][j] > 0)
                {
                    unsigned short tmp = 65535 * (filtered_z[i][j] - min_depth) / (max_depth - min_depth);
                    range_img.at<cv::Vec3s>(i, j) = cv::Vec3s(tmp, tmp, tmp);
                    costs[i][j] = 0;
                    cnts[i][j]++;
                    que.push(i * width + j);
                }
            }
        }

        //int dx[8]{-1, 0, 1, -1, 1, -1, 0, 1};
        //int dy[8]{-1, -1, -1, 0, 0, 1, 1, 1};
        int dx[4]{-1, 1, 0, 0};
        int dy[4]{0, 0, -1, 1};
        while (!que.empty())
        {
            int now = que.front();
            int x = now % width;
            int y = now / width;
            que.pop();

            unsigned short val = range_img.at<cv::Vec3s>(y, x)[0];
            int next_cost = costs[y][x] + 1;
            for (int i = 0; i < 4; i++)
            {
                int toX = x + dx[i];
                int toY = y + dy[i];
                if (toX < 0 || toX >= width || toY < 0 || toY >= height)
                {
                    continue;
                }

                if (costs[toY][toX] < next_cost)
                {
                    continue;
                }

                if (next_cost < costs[toY][toX])
                {
                    que.push(toY * width + toX);
                }

                //unsigned short tmp = range_img.at<unsigned short>(toY, toX);
                //range_img.at<unsigned short>(toY, toX) = (unsigned short)(((int)tmp * cnts[toY][toX] + val * cnts[y][x]) / (cnts[toY][toX] + cnts[y][x]));

                //unsigned short tmp = range_img.at<cv::Vec3s>(toY, toX)[0];
                //tmp = (unsigned short)(((int)tmp * cnts[toY][toX] + val * cnts[y][x]) / (cnts[toY][toX] + cnts[y][x]));

                range_img.at<cv::Vec3s>(toY, toX) = cv::Vec3s(val, val, val);
                costs[toY][toX] = next_cost;
                cnts[toY][toX] += cnts[y][x];
            }
        }
        //cout << "Sample time[ms] = " << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << endl;

        /*
        cv::Ptr<cv::ORB> point_orb = cv::ORB::create();
        vector<cv::KeyPoint> point_key;
        cv::Mat point_descriptor;
        point_orb->detectAndCompute(range_img, cv::Mat(), point_key, point_descriptor);
        cv::Mat point_key_img = cv::Mat::zeros(height, width, CV_16UC3);
        cv::drawKeypoints(range_img, point_key, point_key_img);
        cv::imshow("Keys", point_key_img);

        cv::Ptr<cv::ORB> thermal_orb = cv::ORB::create();
        vector<cv::KeyPoint> thermal_key;
        cv::Mat thermal_descriptor;
        thermal_orb->detectAndCompute(img, cv::Mat(), thermal_key, thermal_descriptor);
        cv::Mat thermal_key_img = cv::Mat::zeros(height, width, CV_8UC3);
        cv::drawKeypoints(img, thermal_key, thermal_key_img);
        cv::imshow("Thermal Keys", thermal_key_img);

        vector<vector<cv::DMatch>> matches;
        cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        matcher.knnMatch(point_descriptor, thermal_descriptor, matches, 2);
        vector<cv::DMatch> good_matches;
        cout << matches.size() << endl;
        for (size_t i = 0; i < matches.size(); i++)
        {
            if (0 < matches[i].size() && matches[i][0].distance < 0.7f * matches[i][1].distance)
            {
                good_matches.push_back(matches[i][0]);
            }
        }
        cv::Mat img_matches;
        cv::drawMatches(range_img, point_key, img, thermal_key, good_matches, img_matches);
        cv::imshow("Matches", img_matches);

        cv::waitKey();
        */
    }

    cv::Mat credibility_img = cv::Mat::zeros(height, width, CV_16UC1);
    {
        cv::Laplacian(range_img, credibility_img, CV_16UC1);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int val = credibility_img.at<unsigned short>(i, j);
                credibility_img.at<unsigned short>(i, j) = (unsigned short)(65535 * exp(-val * val / 2 / sigma_c / sigma_c));
            }
        }
        cv::imshow("b", credibility_img);
        cv::waitKey();
    }

    cv::Mat jbu_img = cv::Mat::zeros(height, width, CV_16UC1);
    // Still slow
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                double coef = 0;
                double val = 0;
                int d0 = img.at<cv::Vec3b>(i, j)[0];
                for (int ii = 0; ii < r; ii++)
                {
                    for (int jj = 0; jj < r; jj++)
                    {
                        int x = jj - r / 2;
                        int y = ii - r / 2;
                        if (i + y < 0 || i + y >= height || j + x < 0 || j + x >= width)
                        {
                            continue;
                        }
                        int d1 = img.at<cv::Vec3b>(i + y, j + x)[0];
                        double tmp = exp(-(x * x + y * y) / 2 / sigma_s / sigma_s) * exp(-(d0 - d1) * (d0 - d1) / 2 / sigma_r / sigma_r) * credibility_img.at<unsigned short>(i + y, j + x);
                        coef += tmp;
                        val += tmp * range_img.at<unsigned short>(i + y, j + x);
                    }
                }
                jbu_img.at<unsigned short>(i, j) = (unsigned short)(val / coef);
            }
        }
        // cv::imshow("c", jbu_img);
    }

    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    vector<vector<double>> interpolated_z(height, vector<double>(width, -1));
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (base_z[i][j] == 0)
            {
                continue;
            }

            double z = (max_depth - min_depth) * jbu_img.at<unsigned short>(i, j) / 65535 + min_depth;
            double x = z * (j - width / 2) / f_x;
            double y = z * (i - height / 2) / f_x;
            interpolated_ptr->points_.emplace_back(x, y, z);
            interpolated_z[i][j] = z;
        }
    }

    double error = 0;
    { // Evaluation
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
        error /= cnt;
        cout << "cannot cnt = " << cannot_cnt - cnt << endl;
        //cout << "Error = " << error << endl;
    }
    cout << "Total time[ms] = " << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << endl;

    if (see_res)
    {
        Eigen::MatrixXd front(4, 4);
        front << 1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, -1, 0,
            0, 0, 0, 1;
        pcd_ptr->Transform(front);
        filtered_ptr->Transform(front);
        interpolated_ptr->Transform(front);

        visualization::DrawGeometries({interpolated_ptr}, "PointCloud", 1600, 900);
    }

    return error;
}

int main(int argc, char *argv[])
{
    //vector<int> data_nos = {550, 1000, 1125, 1260, 1550};
    vector<int> data_nos = {10, 20, 30, 40, 50}; // 02_19_13jo
    //vector<int> data_nos = {700, 1290, 1460, 2350, 3850}; // 02_04_miyanosawa

    for (int i = 0; i < data_nos.size(); i++)
    {
        cout << segmentate(data_nos[i], 91, 46, 1, 19, false) << endl;
    }

    double best_error = 1000;
    double best_sigma_c = 1;
    double best_sigma_s = 1;
    double best_sigma_r = 1;
    int best_r = 1;
    // best params 2020/07/06 sigma_c:91 sigma_s:46 sigma_R:1 r:19

    for (double sigma_c = 1; sigma_c < 100; sigma_c += 10)
    {
        for (double sigma_s = 1; sigma_s < 50; sigma_s += 5)
        {
            for (double sigma_r = 1; sigma_r < 5; sigma_r += 5)
            {
                for (int r = 1; r < 20; r++)
                {
                    double error = 0;
                    for (int i = 0; i < data_nos.size(); i++)
                    {
                        error += segmentate(data_nos[i], sigma_c, sigma_s, sigma_r, r);
                    }

                    if (best_error > error)
                    {
                        best_sigma_c = sigma_c;
                        best_sigma_s = sigma_s;
                        best_sigma_r = sigma_r;
                        best_r = r;
                    }
                }
            }
        }
    }

    cout << "Sigma C = " << best_sigma_c << endl;
    cout << "Sigma S = " << best_sigma_s << endl;
    cout << "Sigma R = " << best_sigma_r << endl;
    cout << "R = " << best_r << endl;
    cout << "Mean error = " << best_error / data_nos.size() << endl;
    return 0;
}