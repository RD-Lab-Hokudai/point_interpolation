#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"

using namespace std;

void generate_depth_image(vector<vector<double>> &target_grid, vector<vector<int>> &target_vs,
                          EnvParams &envParams)
{
    double max_dist = 0;
    for (int i = 0; i < target_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            max_dist = max(max_dist, target_grid[i][j]);
        }
    }

    cv::Mat depth_img = cv::Mat::zeros(target_vs.size(), envParams.width, CV_8UC1);
    cout << target_grid.size() << " " << target_grid[0].size() << endl;
    for (int i = 0; i < target_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            if (target_grid[i][j] <= 0)
            {
                continue;
            }

            double val = min(50.0, target_grid[i][j]);
            depth_img.at<uchar>(i, j) = 255 * val / 50;
            //depth_img.at<uchar>(i, j) = 255 * target_grid[i][j] / max_dist;
        }
    }
    cv::imshow("depth", depth_img);
    //cv::imwrite("depth.png", depth_img);
    cv::waitKey();
}