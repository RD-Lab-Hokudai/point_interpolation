#pragma once
#include <vector>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <time.h>

#include "../models/envParams.cpp"
#include "quality_metrics_OpenCV_2.cpp"

using namespace std;
using namespace open3d;

void evaluate(vector<vector<double>> &target_grid, vector<vector<double>> &original_grid, vector<vector<int>> &target_vs, vector<vector<int>> &original_vs, EnvParams envParams, int layer_cnt, double &ssim, double &mse, double &mre)
{
    cv::Mat target_Mat = cv::Mat::zeros(envParams.height, envParams.width, CV_64FC1);
    cv::Mat original_reproject_Mat = cv::Mat::zeros(original_vs.size(), envParams.width, CV_64FC1);
    cv::Mat target_reproject_Mat = cv::Mat::zeros(original_vs.size(), envParams.width, CV_64FC1);

    for (int i = 0; i < target_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            target_Mat.at<double>(target_vs[i][j], j) = target_grid[i][j];
        }
    }

    for (int i = 0; i < original_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            original_reproject_Mat.at<double>(i, j) = original_grid[i][j];
            target_reproject_Mat.at<double>(i, j) = target_Mat.at<double>(original_vs[i][j], j);
        }
    }

    cout << original_reproject_Mat.rows << endl;
    ssim = qm::ssim(original_reproject_Mat, target_reproject_Mat, 64 / layer_cnt);
    mse = qm::eqm(original_reproject_Mat, target_reproject_Mat);
    mre = qm::mre(original_reproject_Mat, target_reproject_Mat);
}
