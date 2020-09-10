#pragma once
#include <vector>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"

using namespace std;
using namespace open3d;

double restore_pcd(vector<vector<double>> &target_grid, vector<vector<double>> &original_grid, vector<vector<int>> &target_vs, vector<vector<int>> &original_vs, EnvParams envParams, cv::Mat img, shared_ptr<geometry::PointCloud> target_ptr, shared_ptr<geometry::PointCloud> original_ptr)
{
    vector<vector<double>> original_full_grid(envParams.height, vector<double>(envParams.width, 0));
    for (int i = 0; i < original_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            original_full_grid[original_vs[i][j]][j] = original_grid[i][j];
        }
    }

    for (int i = 0; i < target_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            double z = target_grid[i][j];
            double original_z = original_full_grid[target_vs[i][j]][j];
            if (z <= 0) //|| original_zenvParams.height <= 0)
            {
                continue;
            }

            z = min(z, 100.0);
            double x = z * (j - envParams.width / 2) / envParams.f_xy;
            double y = z * (target_vs[i][j] - envParams.height / 2) / envParams.f_xy;
            target_ptr->points_.emplace_back(100 + x, z, -y);

            cv::Vec3b color = img.at<cv::Vec3b>(target_vs[i][j], j);
            target_ptr->colors_.emplace_back(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0);
        }
    }
    for (int i = 0; i < original_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            double original_z = original_grid[i][j];
            if (original_z <= 0)
            {
                continue;
            }

            double original_x = original_z * (j - envParams.width / 2) / envParams.f_xy;
            double original_y = original_z * (original_vs[i][j] - envParams.height / 2) / envParams.f_xy;
            original_ptr->points_.emplace_back(original_x, original_z, -original_y);

            cv::Vec3b color = img.at<cv::Vec3b>(original_vs[i][j], j);
            original_ptr->colors_.emplace_back(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0);
        }
    }
}
