#pragma once
#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"
#include "../utils/UnionFind.cpp"
#include "../utils/SegmentationGraph.cpp"

using namespace std;

void jbu_cv(cv::Mat &target_mat, cv::Mat &base_mat, cv::Mat &target_vs_mat, cv::Mat &base_vs_mat, EnvParams envParams, cv::Mat &img, double sigma_c, double sigma_s, double sigma_r, double r)
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

    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};
    cv::Mat credibilities = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    credibilities.forEach<double>([&](double &now, const int position[]) -> void {
        int y = position[0];
        int x = position[1];

        cv::Vec3b val = 0;
        int cnt = 0;
        for (int k = 0; k < 4; k++)
        {
            int xx = x + dx[k];
            int yy = y + dy[k];
            if (xx < 0 || xx >= target_vs_mat.cols || yy < 0 || yy >= target_vs_mat.rows)
            {
                continue;
            }
            int v = target_vs_mat.at<int>(yy, xx);

            val += img.at<cv::Vec3b>(v, x);
            cnt++;
        }
        val -= cnt * img.at<cv::Vec3b>(target_vs_mat.at<int>(y, x), x);
        //val = 65535 * (val - min_depth) / (max_depth - min_depth);
        now = exp(-cv::norm(val) / 2 / sigma_c / sigma_c);
    });

    // JBU
    cv::Mat jbu_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    jbu_mat.forEach<double>([&](double &now, const int position[]) -> void {
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
                    double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s) * exp(-cv::norm(d0 - d1) / 2 / sigma_r / sigma_r) * credibilities.at<double>(y + dy, x + dx);
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

    double max_dist = 500;
    cv::Mat inverted = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    inverted.forEach<double>([&jbu_mat, &max_dist](double &now, const int position[]) -> void {
        double d = jbu_mat.at<double>(position[0], position[1]);
        if (d > 0)
        {
            now = max_dist - d;
        }
    });

    cv::Mat filled, filled2;
    cv::Mat fix_calibration_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(inverted, filled, cv::MORPH_CLOSE, fix_calibration_kernel);
    cv::Mat full_fill_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31));
    cv::dilate(filled, filled2, full_fill_kernel);
    filled2.forEach<double>([&filled](double &now, const int position[]) -> void {
        double d = filled.at<double>(position[0], position[1]);
        if (d > 0)
        {
            now = d;
        }
    });

    //cv::imshow("D", filled);
    //cv::waitKey();
    /*
    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);
    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);

    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);
    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);
    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);
    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);
*/
    target_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    target_mat.forEach<double>([&filled2, &max_dist](double &now, const int position[]) -> void {
        double d = filled2.at<double>(position[0], position[1]);
        if (d > 0)
        {
            now = max_dist - d;
        }
    });
}