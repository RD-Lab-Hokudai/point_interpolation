#pragma once
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"

using namespace std;

void linear(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
            EnvParams env_params) {
  cv::Mat full_grid =
      cv::Mat::zeros(env_params.height, env_params.width, CV_64FC1);

  for (int j = 0; j < env_params.width; j++) {
    ushort vNext = vs.at<ushort>(0, j);
    for (int k = 0; k < vNext; k++) {
      full_grid.at<double>(k, j) = src_grid.at<double>(0, j);
    }
  }
  cv::imshow("A", full_grid);
  cv::imshow("B", src_grid);
  cv::waitKey();

  for (int i = 0; i + 1 < vs.rows; i++) {
    for (int j = 0; j < vs.cols; j++) {
      double zPrev = src_grid.at<double>(i, j);
      double zNext = src_grid.at<double>(i + 1, j);
      ushort vPrev = vs.at<ushort>(i, j);
      ushort vNext = vs.at<ushort>(i + 1, j);
      double yPrev = zPrev * (vPrev - env_params.height / 2) / env_params.f_xy;
      double yNext = zNext * (vNext - env_params.height / 2) / env_params.f_xy;
      double angle = (zNext - zPrev) / (yNext - yPrev);  // 傾き

      for (int k = 0; vPrev + k < vNext; k++) {
        int v = vPrev + k;
        double tan = (v - env_params.height / 2) / env_params.f_xy;
        double z = (zPrev - angle * yPrev) / (1 - angle * tan);
        full_grid.at<double>(v, j) = z;
      }
    }
  }

  cv::imshow("A", full_grid);
  cv::imshow("B", src_grid);
  cv::waitKey();

  for (int j = 0; j < env_params.width; j++) {
    ushort vPrev = vs.at<ushort>(vs.rows - 1, j);
    for (int k = 0; vPrev + k < env_params.height; k++) {
      int v = vPrev + k;
      full_grid.at<double>(v, j) = src_grid.at<double>(vs.rows - 1, j);
    }
  }

  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst_grid.forEach<double>(
      [&vs, &full_grid](double& now, const int position[]) -> void {
        now = full_grid.at<double>(vs.at<ushort>(position[0], position[1]),
                                   position[1]);
      });
  cv::imshow("A", full_grid);
  cv::waitKey();
}
