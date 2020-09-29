#pragma once
#include <vector>
#include <queue>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"

using namespace std;
using namespace open3d;

void find_neighbors(EnvParams envParams, vector<vector<double>> &grid, vector<vector<int>> &vs, vector<vector<vector<int>>> &neighbors, int neighbor_cnt)
{
    auto start = chrono::system_clock::now();
    neighbors = vector<vector<vector<int>>>(vs.size(), vector<vector<int>>(envParams.width, vector<int>()));
    queue<int> que;
    queue<int> que_idx;
    for (int i = 0; i < vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            if (grid[i][j] > 0)
            {
                que.emplace(i * envParams.width + j);
                que_idx.emplace(i * envParams.width + j);
            }
        }
    }

    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};
    while (!que.empty())
    {
        int val = que.front();
        int idx = que_idx.front();
        que.pop();
        que_idx.pop();

        for (int i = 0; i < 4; i++)
        {
            int x = val % envParams.width + dx[i];
            int y = val / envParams.width + dy[i];

            if (x < 0 || x >= envParams.width || y < 0 || y >= vs.size())
            {
                continue;
            }

            if (neighbors[y][x].size() == neighbor_cnt)
            {
                continue;
            }

            bool duplicate = false;
            for (int j = 0; j < neighbors[y][x].size(); j++)
            {
                if (neighbors[y][x][j] == idx)
                {
                    duplicate = true;
                    break;
                }
            }

            if (duplicate)
            {
                continue;
            }

            neighbors[y][x].emplace_back(idx);
            que.emplace(y * envParams.width + x);
            que_idx.emplace(idx);
        }
    }
    cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;

    cv::Mat img = cv::Mat::zeros(vs.size(), envParams.width, CV_16UC1);
    for (int i = 0; i < vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            img.at<ushort>(i, j) = neighbors[i][j][0];
        }
    }
    cv::imshow("aiueo", img);
    cv::waitKey();

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