#pragma once
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"
#include "linear.cpp"

using namespace std;

void pwas(vector<vector<double>> &target_grid, vector<vector<double>> &base_grid, vector<vector<int>> &target_vs, vector<vector<int>> &base_vs, EnvParams envParams, cv::Mat img)
{
    // Linear interpolation
    vector<vector<double>> linear_grid(target_vs.size(), vector<double>(envParams.width, 0));
    linear(linear_grid, base_grid, target_vs, base_vs, envParams);

    // PWAS
    vector<vector<double>> credibilities(target_vs.size(), vector<double>(envParams.width));
    cv::Mat credibility_img(target_vs.size(), envParams.width, CV_16UC1);
    double sigma_c = 10;
    double sigma_s = 1.6;
    double sigma_r = 19;
    //sigma_r = 0.1;
    double r = 7;

    {
        double min_depth = 1000000;
        double max_depth = 0;
        for (int i = 0; i < target_vs.size(); i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                min_depth = min(min_depth, linear_grid[i][j]);
                max_depth = max(max_depth, linear_grid[i][j]);
            }
        }
        cout << min_depth << " " << max_depth << endl;

        int dx[] = {1, -1, 0, 0};
        int dy[] = {0, 0, 1, -1};
        for (int i = 0; i < target_vs.size(); i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                double val = 0;
                int cnt = 0;
                for (int k = 0; k < 4; k++)
                {
                    int x = j + dx[k];
                    int y = i + dy[k];
                    if (x < 0 || x >= envParams.width || y < 0 || y >= target_vs.size())
                    {
                        continue;
                    }

                    val += linear_grid[y][x];
                    cnt++;
                }
                val -= cnt * linear_grid[i][j];
                //val = 65535 * (val - min_depth) / (max_depth - min_depth);
                credibilities[i][j] = exp(-val * val / 2 / sigma_c / sigma_c);
                //credibility_img.at<ushort>(i, j) = 10000 * credibilities[i][j];
            }
        }
    }
    /*
    cv::imshow("creds", credibility_img);
    cv::waitKey();
    */

    target_grid = vector<vector<double>>(target_vs.size(), vector<double>(envParams.width, 0));
    // Still slow
    {
        for (int i = 0; i < target_vs.size(); i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                double coef = 0;
                double val = 0;
                int v = target_vs[i][j];
                cv::Vec3b d0 = img.at<cv::Vec3b>(v, j);

                for (int ii = 0; ii < r; ii++)
                {
                    for (int jj = 0; jj < r; jj++)
                    {
                        int dy = ii - r / 2;
                        int dx = jj - r / 2;
                        if (i + dy < 0 || i + dy >= target_vs.size() || j + dx < 0 || j + dx >= envParams.width)
                        {
                            continue;
                        }

                        int v1 = target_vs[i + dy][j + dx];
                        cv::Vec3b d1 = img.at<cv::Vec3b>(v1, j + dx);
                        double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s) * exp(-cv::norm(d0 - d1) / 2 / sigma_r / sigma_r) * credibilities[i + dy][j + dx];
                        //cout << tmp << endl;
                        val += tmp * linear_grid[i + dy][j + dx];
                        coef += tmp;
                    }
                }
                if (coef > 0)
                {
                    target_grid[i][j] = val / coef;
                }
            }
        }
    }
}