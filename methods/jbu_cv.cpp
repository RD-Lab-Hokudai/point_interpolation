#pragma once
#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"
#include "../utils/UnionFind.cpp"
#include "../utils/SegmentationGraph.cpp"

using namespace std;

void jbu_cv(cv::Mat &target_mat, cv::Mat &base_mat, cv::Mat &target_vs_mat, cv::Mat &base_vs_mat, EnvParams envParams, cv::Mat &img, double sigma_s, double sigma_r, double r)
{
    cv::Mat full_mat = cv::Mat::zeros(envParams.height, envParams.width, CV_64FC1);
    base_mat.forEach<double>([&full_mat, &base_vs_mat](double &now, const int position[]) -> void {
        int v = base_vs_mat.at<int>(position[0], position[1]);
        full_mat.at<double>(v, position[1]) = now;
    });
    cv::Mat low_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    low_mat.forEach<double>([&full_mat, &target_vs_mat](double &now, const int position[]) -> void {
        int v = target_vs_mat.at<int>(position[0], position[1]);
        now = full_mat.at<double>(v, position[1]);
    });

    // JBU
    target_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    target_mat.forEach<double>([&](double &now, const int position[]) -> void {
        int y = position[0];
        int x = position[1];
        double coef = 0;
        double val = 0;
        double src_val = base_mat.at<double>(y, x);
        if (src_val > 0)
        {
            now = src_val;
        }
        else
        {
            int v = target_vs_mat.at<int>(y, x);
            cv::Vec3b d0 = img.at<cv::Vec3b>(v, x);

            for (int ii = 0; ii < r; ii++)
            {
                for (int jj = 0; jj < r; jj++)
                {
                    int dy = ii - r / 2;
                    int dx = jj - r / 2;
                    if (y + dy < 0 || y + dy >= target_vs_mat.rows || x + dx < 0 || x + dx >= target_vs_mat.cols)
                    {
                        continue;
                    }

                    int v1 = target_vs_mat.at<int>(y + dy, x + dx);
                    double neighbor_val = base_mat.at<double>(y + dy, x + dx);
                    if (neighbor_val <= 0)
                    {
                        continue;
                    }

                    cv::Vec3b d1 = img.at<cv::Vec3b>(v1, x + dx);
                    double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s) * exp(-cv::norm(d0 - d1) / 2 / sigma_r / sigma_r);
                    val += tmp * neighbor_val;
                    coef += tmp;
                }
            }
            if (coef > 0.0 /* some threshold */)
            {
                now = val / coef;
            }
        }
    });
}