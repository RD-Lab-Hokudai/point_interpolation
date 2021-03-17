#pragma once

#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"

using namespace std;

cv::Mat generateDiamondKernel(int kernel_size) {
  cv::Mat kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_8UC1);
  int center = kernel_size / 2;
  kernel.forEach<uchar>(
      [center, kernel_size](uchar& now, const int position[]) -> void {
        int dist = abs(center - position[0]) + abs(center - position[1]);

        if (dist <= center) {
          now = 1;
        }
      });
  return kernel;
}

void ip_basic(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
              EnvParams env_params) {
  double max_dist = 500;
  cv::Mat inverted = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  inverted.forEach<double>(
      [&src_grid, &max_dist](double& now, const int position[]) -> void {
        double d = src_grid.at<double>(position[0], position[1]);
        if (d > 0) {
          now = max_dist - d;
        }
      });

  cv::Mat dilate_kernel = generateDiamondKernel(5);

  cv::Mat dilated;
  cv::dilate(inverted, dilated, dilate_kernel);

  cv::Mat closed1, filled1;
  cv::Mat close_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(dilated, closed1, cv::MORPH_CLOSE, close_kernel);
  cv::Mat full_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  cv::dilate(closed1, filled1, full_kernel);
  filled1.forEach<double>(
      [&closed1](double& now, const int position[]) -> void {
        double d = closed1.at<double>(position[0], position[1]);
        if (d > 0) {
          now = d;
        }
      });

  for (int j = 0; j < vs.cols; j++) {
    int top = vs.rows;
    for (int i = 0; i < vs.rows; i++) {
      double val = filled1.at<double>(i, j);
      if (val > 0) {
        top = i;
        break;
      }
    }
    if (top == vs.rows) {
      continue;
    }

    double fill_val = filled1.at<double>(top, j);
    for (int i = 0; i < top; i++) {
      filled1.at<double>(i, j) = fill_val;
    }
  }
  cv::Mat filled2;
  cv::Mat full_fill_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31));
  cv::dilate(filled1, filled2, full_fill_kernel);
  filled2.forEach<double>(
      [&filled1](double& now, const int position[]) -> void {
        double d = filled1.at<double>(position[0], position[1]);
        if (d > 0) {
          now = d;
        }
      });

  // cv::Mat blured;
  // cv::GaussianBlur(filled2, blured, cv::Size(5, 5), 0);

  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst_grid.forEach<double>(
      [&filled2, &max_dist](double& now, const int position[]) -> void {
        double d = filled2.at<double>(position[0], position[1]);
        if (d > 0) {
          now = max_dist - d;
        }
      });
}