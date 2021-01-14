#pragma once
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"

using namespace std;

// kernel_size must to be odd number
cv::Mat generateDiamondKernel(int kernel_size)
{
    cv::Mat kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_8UC1);
    int center = kernel_size / 2;
    kernel.forEach<uchar>([center, kernel_size](uchar &now, const int position[]) -> void {
        int dist = abs(center - position[0]) + abs(center - position[1]);

        if (dist <= center)
        {
            now = 1;
        }
    });
    return kernel;
}

void ip_basic(vector<vector<double>> &target_grid, vector<vector<double>> &base_grid, vector<vector<int>> &target_vs, vector<vector<int>> &base_vs, EnvParams envParams)
{
    double max_dist = 500;
    cv::Mat full_inverted = cv::Mat::zeros(envParams.height, envParams.width, CV_64FC1);
    for (int i = 0; i < base_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            int v = base_vs[i][j];
            if (base_grid[i][j] > 0)
            {
                full_inverted.at<double>(v, j) = max_dist - base_grid[i][j];
            }
        }
    }
    //cv::imshow("inv", full_inverted);

    cv::Mat inverted = cv::Mat::zeros(target_vs.size(), envParams.width, CV_64FC1);
    for (int i = 0; i < target_vs.size(); i++)
    {
        double *now = &inverted.at<double>(i, 0);
        for (int j = 0; j < envParams.width; j++)
        {
            int v = target_vs[i][j];
            *now = full_inverted.at<double>(v, j);
            now++;
        }
    }

    cv::Mat depth;
    cv::Mat dilate_kernel = generateDiamondKernel(5);
    cv::Mat dilated;
    cv::dilate(inverted, dilated, dilate_kernel);

    cv::Mat closed, fulled;
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(dilated, closed, cv::MORPH_CLOSE, close_kernel);
    cv::Mat full_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::dilate(closed, depth, full_kernel);
    depth.forEach<double>([&closed](double &now, const int position[]) -> void {
        double d = closed.at<double>(position[0], position[1]);
        if (d > 0)
        {
            now = d;
        }
    });

    /*
    cv::Mat full_fill_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31));
    cv::dilate(depth, fulled, full_fill_kernel);
    depth.forEach<double>([&fulled](double &now, const int position[]) -> void {
        double d = fulled.at<double>(position[0], position[1]);
        if (now == 0)
        {
            now = d;
        }
    });
    */

    target_grid = vector<vector<double>>(target_vs.size(), vector<double>(envParams.width, 0));
    depth.forEach<double>([&target_grid, max_dist](double &now, const int position[]) -> void {
        if (now > 0)
        {
            target_grid[position[0]][position[1]] = max_dist - now;
        }
    });
}