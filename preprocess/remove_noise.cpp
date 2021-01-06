#pragma once
#include <vector>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace open3d;

void remove_noise(vector<vector<double>> &src, vector<vector<double>> &dst, vector<vector<int>> &target_vs, EnvParams envParams, double rad_coef = 0.001)
{
    auto ptr = make_shared<geometry::PointCloud>();
    for (int i = 0; i < target_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            double z = src[i][j];
            if (z <= 0)
            {
                continue;
            }

            double x = z * (j - envParams.width / 2) / envParams.f_xy;
            double y = z * (target_vs[i][j] - envParams.height / 2) / envParams.f_xy;
            ptr->points_.emplace_back(x, y, z);
        }
    }
    auto kdtree = make_shared<geometry::KDTreeFlann>(*ptr);
    vector<vector<double>> full_grid(envParams.height, vector<double>(envParams.width, 0));
    for (int i = 0; i < ptr->points_.size(); i++)
    {
        double x = ptr->points_[i][0];
        double y = ptr->points_[i][1];
        double z = ptr->points_[i][2];
        double distance2 = x * x + y * y + z * z;

        //探索半径：係数*(距離)^2
        double radius = rad_coef * distance2;

        //最も近い点を探索し，半径r以内にあるか判定
        vector<int> indexes(2);
        vector<double> dists(2);
        kdtree->SearchKNN(ptr->points_[i], 2, indexes, dists);

        //radiusを超えない範囲に近傍点があれば残す
        if (dists[1] <= radius)
        {
            int u = x / z * envParams.f_xy + envParams.width / 2;
            int v = y / z * envParams.f_xy + envParams.height / 2;
            full_grid[v][u] = z;
        }
    }

    dst = vector<vector<double>>(target_vs.size(), vector<double>(envParams.width, 0));
    for (int i = 0; i < target_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            dst[i][j] = full_grid[target_vs[i][j]][j];
        }
    }
}

void remove_noise_cv(cv::Mat &src, cv::Mat &dst, cv::Mat &target_vs_mat, EnvParams envParams, double rad_coef = 0.001)
{
    auto ptr = make_shared<geometry::PointCloud>();
    src.forEach<double>([&ptr, &envParams, &target_vs_mat](double &now, const int position[]) -> void {
        if (now <= 0)
        {
            return;
        }

        double x = now * (position[1] - envParams.width / 2) / envParams.f_xy;
        double y = now * (target_vs_mat.at<int>(position[0], position[1]) - envParams.height / 2) / envParams.f_xy;
        ptr->points_.emplace_back(x, y, now);
    });
    auto kdtree = make_shared<geometry::KDTreeFlann>(*ptr);
    cv::Mat full_grid = cv::Mat::zeros(envParams.height, envParams.width, CV_64FC1);
    for (int i = 0; i < ptr->points_.size(); i++)
    {
        double x = ptr->points_[i][0];
        double y = ptr->points_[i][1];
        double z = ptr->points_[i][2];
        double distance2 = x * x + y * y + z * z;

        //探索半径：係数*(距離)^2
        double radius = rad_coef * distance2;

        //最も近い点を探索し，半径r以内にあるか判定
        vector<int> indexes(2);
        vector<double> dists(2);
        kdtree->SearchKNN(ptr->points_[i], 2, indexes, dists);

        //radiusを超えない範囲に近傍点があれば残す
        if (dists[1] <= radius)
        {
            int u = x / z * envParams.f_xy + envParams.width / 2;
            int v = y / z * envParams.f_xy + envParams.height / 2;
            full_grid.at<double>(v, u) = z;
        }
    }

    dst = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    dst.forEach<double>([&](double &now, const int position[]) -> void {
        now = full_grid.at<double>(target_vs_mat.at<int>(position[0], position[1]), position[1]);
    });
}

/*
void remove_noise_2d(vector<vector<double>> &src, vector<vector<double>> &dst, vector<vector<int>> &target_vs, EnvParams envParams, double rad_coef = 0.001)
{
    int r = 101;
    dst = vector<vector<double>>(target_vs.size(), vector<double>(target_vs[0].size(), 0));
    for (int i = 0; i < target_vs.size(); i++)
    {
        for (int j = 0; j < target_vs[0].size(); j++)
        {
            double z = src[i][j];
            if (z <= 0)
            {
                continue;
            }
            int v = target_vs[i][j];
            double x = z * (j - envParams.width / 2) / envParams.f_xy;
            double y = z * (v - envParams.height / 2) / envParams.f_xy;

            double dist2 = x * x + y * y + z * z;
            double radius = rad_coef * dist2;

            int ok_cnt = 0;
            for (int ii = 0; ii < r; ii++)
            {
                for (int jj = 0; jj < r; jj++)
                {
                    int i_tmp = i + ii - r / 2;
                    int j_tmp = j + jj - r / 2;
                    if (i_tmp < 0 || i_tmp >= target_vs.size() || j_tmp < 0 || j_tmp >= target_vs[0].size())
                    {
                        continue;
                    }
                    if (i_tmp == i && j_tmp == j)
                    {
                        continue;
                    }

                    double z_tmp = src[i_tmp][j_tmp];
                    if (z_tmp <= 0)
                    {
                        continue;
                    }
                    double x_tmp = z_tmp * (j_tmp - envParams.width / 2) / envParams.f_xy;
                    int v_tmp = target_vs[i_tmp][j_tmp];
                    double y_tmp = z_tmp * (v_tmp - envParams.height / 2) / envParams.f_xy;
                    double distance2 = (x - x_tmp) * (x - x_tmp) + (y - y_tmp) * (y - y_tmp) + (z - z_tmp) * (z - z_tmp);
                    if (distance2 <= radius * radius)
                    {
                        ok_cnt++;
                    }
                }
            }

            if (ok_cnt > 0)
            {
                dst[i][j] = z;
            }
        }
    }
}

void remove_noise_2d_cv(cv::Mat &src_mat, cv::Mat &dst_mat, cv::Mat &target_vs_mat, EnvParams envParams, double rad_coef = 0.001)
{
    int r = 5;
    dst_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    dst_mat.forEach<double>([&](double &now, const int position[]) -> void {
        int i = position[0];
        int j = position[1];
        double z = src_mat.at<double>(i, j);
        if (z <= 0)
        {
            return;
        }
        double x = z * (j - envParams.width / 2) / envParams.f_xy;
        int v = target_vs_mat.at<int>(i, j);
        double y = z * (v - envParams.height / 2) / envParams.f_xy;

        double dist2 = x * x + y * y + z * z;
        double radius = rad_coef * rad_coef * dist2;

        int ok_cnt = 0;
        for (int ii = 0; ii < r; ii++)
        {
            for (int jj = 0; jj < r; jj++)
            {
                int i_tmp = i + ii - r / 2;
                int j_tmp = j + jj - r / 2;
                if (i_tmp < 0 || i_tmp >= target_vs_mat.rows || j_tmp < 0 || j_tmp >= target_vs_mat.cols)
                {
                    continue;
                }
                if (i_tmp == 0 && j_tmp == 0)
                {
                    continue;
                }

                double z_tmp = src_mat.at<double>(i_tmp, j_tmp);
                if (z_tmp <= 0)
                {
                    continue;
                }
                double x_tmp = z_tmp * (j_tmp - envParams.width / 2) / envParams.f_xy;
                int v_tmp = target_vs_mat.at<int>(i_tmp, j_tmp);
                double y_tmp = z_tmp * (v_tmp - envParams.height / 2) / envParams.f_xy;
                double distance2 = (x - x_tmp) * (x - x_tmp) + (y - y_tmp) * (y - y_tmp) + (z - z_tmp) * (z - z_tmp);
                if (distance2 <= radius * radius)
                {
                    ok_cnt++;
                }
            }
        }

        if (ok_cnt > 0)
        {
            now = z;
        }
    });
}
*/