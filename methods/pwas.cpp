#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"

using namespace std;

void pwas(const cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs, cv::Mat& img,
          double sigma_c, double sigma_s, double sigma_r, double r) {
  cv::Mat credibilities = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);

  int dx[] = {1, -1, 0, 0};
  int dy[] = {0, 0, 1, -1};
  credibilities.forEach<double>([&](double& now, const int position[]) -> void {
    cv::Vec3b val = 0;
    int cnt = 0;
    for (int k = 0; k < 4; k++) {
      int x = position[1] + dx[k];
      int y = vs.at<ushort>(position[0] + dy[k], x);
      if (x < 0 || x >= vs.cols || y < 0 || y >= vs.rows) {
        continue;
      }

      val += img.at<cv::Vec3b>(y, x);
      cnt++;
    }
    val -= cnt * img.at<cv::Vec3b>(vs.at<ushort>(position[0], position[1]),
                                   position[1]);
    now = exp(-cv::norm(val) / 2 / sigma_c / sigma_c);
  });

  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst_grid.forEach<double>([&](double& now, const int position[]) -> void {
    double coef = 0;
    double val = 0;

    // すでに点が与えられているならそれを使う
    double org_val = src_grid.at<double>(position[0], position[1]);
    if (now > 0) {
      now = org_val;
      return;
    }

    cv::Vec3b d0 =
        img.at<cv::Vec3b>(vs.at<ushort>(position[0], position[1]), position[1]);

    for (int ii = 0; ii < r; ii++) {
      for (int jj = 0; jj < r; jj++) {
        int dy = ii - r / 2;
        int dx = jj - r / 2;
        int tmp_y = position[0] + dy;
        int tmp_x = position[1] + dx;
        if (tmp_y < 0 || tmp_y >= vs.rows || tmp_x < 0 || tmp_x >= vs.cols) {
          continue;
        }

        if (src_grid.at<double>(tmp_y, tmp_x) <= 0) {
          continue;
        }

        cv::Vec3b d1 = img.at<cv::Vec3b>(vs.at<ushort>(tmp_y, tmp_x), tmp_x);
        double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s) *
                     exp(-cv::norm(d0 - d1) / 2 / sigma_r / sigma_r) *
                     credibilities.at<double>(tmp_y, tmp_x);
        val += tmp * src_grid.at<double>(tmp_y, tmp_x);
        coef += tmp;
      }
    }
    if (coef > 0) {
      now = val / coef;
    }
  });
}