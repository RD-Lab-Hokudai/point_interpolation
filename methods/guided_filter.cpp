#pragma once
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>

#include "../models/envParams.cpp"

using namespace std;

void calc_exist_means(const cv::Mat &mat, cv::Mat &mean_mat, int &r, const cv::Mat mask_mat)
{
    cv::Mat mask_mean;
    cv::blur(mask_mat, mask_mean, cv::Size(r, r));

    cv::blur(mat, mean_mat, cv::Size(r, r));
    mean_mat.forEach<double>([&mask_mean, &r](double &now, const int position[]) -> void {
        double mask = mask_mean.at<double>(position[0], position[1]);
        if (mask * r * r > 0.5)
        {
            now /= mask;
        }
        else
        {
            now = 0;
        }
    });
    //mean_mat /= mask_mean;

    /*
    mean_mat = cv::Mat::zeros(mat.rows, mat.cols, CV_64FC1);
    mean_mat.forEach<double>([&mat, &mask_mat, &r](double &now, const int position[]) -> void {
        int x = position[1];
        int y = position[0];
        int left = max(x - r / 2, 0);
        int right = min(x + r / 2 + 1, mat.cols);
        int top = max(y - r / 2, 0);
        int bottom = min(y + r / 2 + 1, mat.rows);
        cv::Mat range = mat(cv::Range(top, bottom), cv::Range(left, right));
        cv::Mat mask_range = mask_mat(cv::Range(top, bottom), cv::Range(left, right));
        auto scalar = cv::mean(range, mask_range > 0);
        now = scalar[0];
    });
    */
}

void guided_filter(cv::Mat &target_mat, cv::Mat &base_mat, cv::Mat &target_vs_mat, cv::Mat &base_vs_mat, EnvParams envParams, cv::Mat img)
{
    cv::Mat full_mat = cv::Mat::zeros(envParams.height, envParams.width, CV_64FC1);
    base_mat.forEach<double>([&full_mat, &base_vs_mat](double &now, const int position[]) -> void {
        int v = base_vs_mat.at<int>(position[0], position[1]);
        full_mat.at<double>(v, position[1]) = now;
    });
    double max_depth = 0;
    cv::Mat low_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    low_mat.forEach<double>([&full_mat, &target_vs_mat, &max_depth](double &now, const int position[]) -> void {
        int v = target_vs_mat.at<int>(position[0], position[1]);
        now = (float)full_mat.at<double>(v, position[1]);
        max_depth = max(max_depth, now);
    });

    cv::Mat gray_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    gray_mat.forEach<double>([&img, &target_vs_mat, &low_mat](double &now, const int position[]) -> void {
        int v = target_vs_mat.at<int>(position[0], position[1]);
        if (low_mat.at<double>(position[0], position[1]) > 0)
        {
            cv::Vec3b c = img.at<cv::Vec3b>(v, position[1]);
            now = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
            now /= 256;
        }
    });

    int r = 11;
    double eps = 256 * 0.3;
    cv::Mat mask_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    mask_mat.setTo(1, low_mat > 0);
    cv::Mat guide_mean, guide_square_mean, guide_variance;
    calc_exist_means(gray_mat, guide_mean, r, mask_mat);
    calc_exist_means(gray_mat.mul(gray_mat), guide_square_mean, r, mask_mat);
    guide_variance = guide_square_mean - guide_mean.mul(guide_mean);
    cv::Mat depth_mean;
    cv::Mat guide_depth_mean, covariance;
    calc_exist_means(low_mat, depth_mean, r, mask_mat);
    calc_exist_means(gray_mat.mul(low_mat), guide_depth_mean, r, mask_mat);
    covariance = guide_depth_mean - guide_mean.mul(depth_mean);
    cv::Mat a = covariance / (guide_variance + eps);
    cv::Mat b = depth_mean - a.mul(guide_mean);
    cv::Mat mean_a, mean_b;
    cv::blur(a, mean_a, cv::Size(r, r));
    cv::blur(b, mean_b, cv::Size(r, r));

    target_mat = mean_a.mul(gray_mat) + mean_b;
}