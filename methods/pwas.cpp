#pragma once
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"
#include "linear.cpp"

using namespace std;

void pwas(vector<vector<double>> &target_grid, vector<vector<double>> &base_grid, vector<vector<int>> &target_vs, vector<vector<int>> &base_vs, EnvParams envParams, cv::Mat img, double sigma_c, double sigma_s, double sigma_r, double r)
{
    vector<vector<double>> full_grid(envParams.height, vector<double>(envParams.width, 0));
    for (int i = 0; i < base_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            full_grid[base_vs[i][j]][j] = base_grid[i][j];
        }
    }

    // PWAS
    vector<vector<double>> credibilities(target_vs.size(), vector<double>(envParams.width));
    cv::Mat credibility_img(target_vs.size(), envParams.width, CV_16UC1);

    {

        int dx[] = {1, -1, 0, 0};
        int dy[] = {0, 0, 1, -1};
        for (int i = 0; i < target_vs.size(); i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                cv::Vec3b val = 0;
                int cnt = 0;
                for (int k = 0; k < 4; k++)
                {
                    int x = j + dx[k];
                    int y = target_vs[i][j] + dy[k];
                    if (x < 0 || x >= envParams.width || y < 0 || y >= envParams.height)
                    {
                        continue;
                    }

                    val += img.at<cv::Vec3b>(y, x);
                    cnt++;
                }
                val -= cnt * img.at<cv::Vec3b>(target_vs[i][j], j);
                //val = 65535 * (val - min_depth) / (max_depth - min_depth);
                credibilities[i][j] = exp(-cv::norm(val) / 2 / sigma_c / sigma_c);
                //credibility_img.at<ushort>(i, j) = 100000 * credibilities[i][j];
            }
        }
    }
    /*
    cv::imshow("creds", credibility_img);
    cv::waitKey();
*/

    target_grid = vector<vector<double>>(target_vs.size(), vector<double>(envParams.width, 0));
    // Still slow??
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
                        if (full_grid[v1][j + dx] <= 0)
                        {
                            continue;
                        }

                        cv::Vec3b d1 = img.at<cv::Vec3b>(v1, j + dx);
                        double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s) * exp(-cv::norm(d0 - d1) / 2 / sigma_r / sigma_r) * credibilities[i + dy][j + dx];
                        val += tmp * full_grid[v1][j + dx];
                        coef += tmp;
                        //val += full_grid[v1][j + dx];
                        //coef++;
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