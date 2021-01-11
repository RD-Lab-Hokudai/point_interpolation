#pragma once
#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"
#include "../utils/UnionFind.cpp"
#include "../utils/SegmentationGraph.cpp"

using namespace std;

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

void ip_basic_cv(cv::Mat &target_mat, cv::Mat &base_mat, cv::Mat &target_vs_mat, cv::Mat &base_vs_mat, EnvParams envParams)
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

    double max_dist = 500;
    cv::Mat inverted = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    inverted.forEach<double>([&low_mat, &max_dist](double &now, const int position[]) -> void {
        double d = low_mat.at<double>(position[0], position[1]);
        if (d > 0)
        {
            now = max_dist - d;
        }
    });

    cv::Mat dilate_kernel = generateDiamondKernel(5);

    cv::Mat dilated;
    cv::dilate(inverted, dilated, dilate_kernel);

    cv::Mat closed1, filled1;
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(dilated, closed1, cv::MORPH_CLOSE, close_kernel);
    cv::Mat full_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::dilate(closed1, filled1, full_kernel);
    filled1.forEach<double>([&closed1](double &now, const int position[]) -> void {
        double d = closed1.at<double>(position[0], position[1]);
        if (d > 0)
        {
            now = d;
        }
    });

    for (int j = 0; j < target_vs_mat.cols; j++)
    {
        int top = target_vs_mat.rows;
        for (int i = 0; i < target_vs_mat.rows; i++)
        {
            double val = filled1.at<double>(i, j);
            if (val > 0)
            {
                top = i;
                break;
            }
        }
        if (top == target_vs_mat.rows)
        {
            continue;
        }

        double fill_val = filled1.at<double>(top, j);
        for (int i = 0; i < top; i++)
        {
            filled1.at<double>(i, j) = fill_val;
        }
    }
    cv::Mat filled2;
    cv::Mat full_fill_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31));
    cv::dilate(filled1, filled2, full_fill_kernel);
    filled2.forEach<double>([&filled1](double &now, const int position[]) -> void {
        double d = filled1.at<double>(position[0], position[1]);
        if (d > 0)
        {
            now = d;
        }
    });

    //cv::Mat blured;
    //cv::GaussianBlur(filled2, blured, cv::Size(5, 5), 0);

    target_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    target_mat.forEach<double>([&filled2, &max_dist](double &now, const int position[]) -> void {
        double d = filled2.at<double>(position[0], position[1]);
        if (d > 0)
        {
            now = max_dist - d;
        }
    });
}