#pragma once
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>

#include "../models/env_params.cpp"

using namespace std;

void calc_exist_means(const cv::Mat& mat, cv::Mat& mean_mat, int& r,
                      const cv::Mat mask_mat) {
  cv::Mat mask_mean;
  cv::blur(mask_mat, mask_mean, cv::Size(r, r));

  cv::blur(mat, mean_mat, cv::Size(r, r));
  mean_mat.forEach<double>(
      [&mask_mean, &r](double& now, const int position[]) -> void {
        double mask = mask_mean.at<double>(position[0], position[1]);
        if (mask * r * r > 0.5) {
          now /= mask;
        } else {
          now = 0;
        }
      });
}

void guided_filter(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
                   EnvParams env_params, cv::Mat img) {
  cv::Mat gray = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  gray.forEach<double>(
      [&img, &vs, &src_grid](double& now, const int position[]) -> void {
        int v = vs.at<ushort>(position[0], position[1]);
        if (src_grid.at<double>(position[0], position[1]) <= 0) {
          return;
        }
        cv::Vec3b c = img.at<cv::Vec3b>(v, position[1]);
        now = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
        now /= 256;
      });

  int r = 11;
  double eps = 256 * 0.3;
  cv::Mat mask = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  mask.setTo(1, src_grid > 0);
  cv::Mat guide_mean, guide_square_mean, guide_variance;
  calc_exist_means(gray, guide_mean, r, mask);
  calc_exist_means(gray.mul(gray), guide_square_mean, r, mask);
  guide_variance = guide_square_mean - guide_mean.mul(guide_mean);
  cv::Mat depth_mean;
  cv::Mat guide_depth_mean, covariance;
  calc_exist_means(src_grid, depth_mean, r, mask);
  calc_exist_means(gray.mul(src_grid), guide_depth_mean, r, mask);
  covariance = guide_depth_mean - guide_mean.mul(depth_mean);
  cv::Mat a = covariance / (guide_variance + eps);
  cv::Mat b = depth_mean - a.mul(guide_mean);
  cv::Mat mean_a, mean_b;
  cv::blur(a, mean_a, cv::Size(r, r));
  cv::blur(b, mean_b, cv::Size(r, r));

  dst_grid = mean_a.mul(gray) + mean_b;
}