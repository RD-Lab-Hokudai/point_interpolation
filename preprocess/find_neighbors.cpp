#pragma once
#include <vector>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"

using namespace std;
using namespace open3d;

void find_neighbors(EnvParams envParams, vector<vector<double>> &grid, vector<vector<int>> &vs, vector<vector<vector<int>>> &neighbors)
{
    auto pcd_ptr = make_shared<geometry::PointCloud>();
    for (int i = 0; i < vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            double z = grid[i][j];
            if (z <= 0)
            {
                continue;
            }

            double x = z * (j - envParams.width / 2) / envParams.f_xy;
            double y = z * (vs[i][j] - envParams.height / 2) / envParams.f_xy;
            pcd_ptr->points_.emplace_back(x, z, -y);
        }
    }

    visualization::DrawGeometries({pcd_ptr});
}