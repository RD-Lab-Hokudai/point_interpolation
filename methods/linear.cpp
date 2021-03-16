#pragma once
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"

using namespace std;

void linear(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
            EnvParams env_params) {
  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);

  // Horizontal interpolation
  for (int i = 0; i < vs.rows; i++) {
    double* row = src_grid.ptr<double>(i);
    double* dst_row = dst_grid.ptr<double>(i);

    // Left
    for (int j = 0; j < vs.cols; j++) {
      if (row[j] <= 1e-9) {
        continue;
      }
      for (int jj = 0; jj <= j; jj++) {
        dst_row[jj] = row[j];
      }
      break;
    }

    // Right
    for (int j = vs.cols - 1; j >= 0; j--) {
      if (row[j] <= 1e-9) {
        continue;
      }

      for (int jj = j; jj < vs.cols; jj++) {
        dst_row[jj] = row[j];
      }
      break;
    }

    int prev_u = 0;
    for (int j = 1; j < vs.cols; j++) {
      if (dst_row[j] <= 1e-9) {
        continue;
      }

      double prev_z = dst_row[prev_u];
      double prev_x =
          prev_z * (prev_u - env_params.width / 2) / env_params.f_xy;
      double next_z = dst_row[j];
      double next_x = next_z * (j - env_params.width / 2) / env_params.f_xy;
      for (int jj = prev_u; jj <= j; jj++) {
        double angle = (next_z - prev_z) / (next_x - prev_x);
        double tan = (jj - env_params.width / 2) / env_params.f_xy;
        double z = (prev_z - angle * prev_x) / (1 - tan * angle);
        dst_row[jj] = z;
      }
      prev_u = j;
    }
  }

  // Vertical interpolation
  for (int j = 0; j < vs.cols; j++) {
    // Up
    for (int i = 0; i < vs.rows; i++) {
      double now = dst_grid.at<double>(i, j);
      if (now <= 1e-9) {
        continue;
      }
      for (int ii = 0; ii <= i; ii++) {
        dst_grid.at<double>(ii, j) = now;
      }
      break;
    }

    // Down
    for (int i = vs.rows - 1; i >= 0; i--) {
      double now = dst_grid.at<double>(i, j);
      if (now <= 1e-9) {
        continue;
      }

      for (int ii = i; ii < vs.rows; ii++) {
        dst_grid.at<double>(ii, j) = now;
      }
      break;
    }

    int prev_i = 0;
    for (int i = 1; i < vs.rows; i++) {
      double now = dst_grid.at<double>(i, j);
      if (now <= 1e-9) {
        continue;
      }

      ushort prev_v = vs.at<ushort>(prev_i, j);
      ushort next_v = vs.at<ushort>(i, j);
      if (prev_v >= next_v) {
        continue;
      }

      double prev_z = dst_grid.at<double>(prev_i, j);
      double prev_y =
          prev_z * (prev_v - env_params.height / 2) / env_params.f_xy;
      double next_z = now;
      double next_y =
          next_z * (next_v - env_params.height / 2) / env_params.f_xy;
      for (int ii = prev_i; ii <= i; ii++) {
        ushort now_v = vs.at<ushort>(ii, j);
        double angle = (next_z - prev_z) / (next_y - prev_y);
        double tan = (now_v - env_params.height / 2) / env_params.f_xy;
        double z = (prev_z - angle * prev_y) / (1 - tan * angle);
        dst_grid.at<double>(ii, j) = z;
      }
      prev_i = i;
    }
  }
  cv::imshow("A", dst_grid);
  cv::waitKey();
}
