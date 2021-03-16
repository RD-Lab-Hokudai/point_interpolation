#pragma once

#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"
#include "../utils/SegmentationGraph.cpp"
#include "../utils/UnionFind.cpp"

using namespace std;

void ext_jbu(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
             UnionFind& color_segments, EnvParams& vsenv_params,
             double color_segment_k, double sigma_s, int r, double coef_s) {
  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst_grid.forEach<double>([&](double& now, const int position[]) -> void {
    int y = position[0];
    int x = position[1];
    double coef = 0;
    double val = 0;

    // すでに点が与えられているならそれを使う
    double src_val = src_grid.at<double>(y, x);
    if (src_val > 0) {
      now = src_val;
      return;
    }

    int v = vs.at<ushort>(y, x);
    int r0 = color_segments.root(v * vsenv_params.width + x);

    for (int ii = 0; ii < r; ii++) {
      for (int jj = 0; jj < r; jj++) {
        int dy = ii - r / 2;
        int dx = jj - r / 2;
        if (y + dy < 0 || y + dy >= vs.rows || x + dx < 0 ||
            x + dx >= vs.cols) {
          continue;
        }

        double neighbor_val = src_grid.at<double>(y + dy, x + dx);
        if (neighbor_val <= 0) {
          continue;
        }

        int v1 = vs.at<ushort>(y + dy, x + dx);
        int r1 = color_segments.root(v1 * vsenv_params.width + x + dx);
        double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s);
        if (r1 != r0) {
          tmp *= coef_s;
        }
        val += tmp * neighbor_val;
        coef += tmp;
      }
    }

    /* Bigger threshold will remove noises */
    if (coef > 1e-9) {
      now = val / coef;
    }
  });
}

void original(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
              EnvParams& env_params, cv::Mat& img, double color_segment_k,
              double sigma_s, int r, double coef_s) {
  shared_ptr<UnionFind> color_segments;
  Graph graph(&img);
  color_segments = graph.segmentate(color_segment_k);
  ext_jbu(src_grid, dst_grid, vs, *color_segments, env_params, color_segment_k,
          sigma_s, r, coef_s);

  // 必要に応じて複数回実行
  /*
  cv::Mat twice_grid;
  ext_jbu(dst_grid, twice_grid, vs, *color_segments, env_params,
          color_segment_k, sigma_s, r, coef_s);
  dst_grid = twice_grid;
  */

  // 必要に応じてモルフォロジー処理
  /*
  double max_dist = 500;
  cv::Mat inverted = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  inverted.forEach<double>(
      [&dst_grid, &max_dist](double& now, const int position[]) -> void {
        double d = dst_grid.at<double>(position[0], position[1]);
        if (d > 0) {
          now = max_dist - d;
        }
      });

  cv::Mat filled, filled2;
  cv::Mat fix_calibration_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  cv::morphologyEx(inverted, filled, cv::MORPH_CLOSE, fix_calibration_kernel);
  cv::Mat full_fill_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31));
  cv::dilate(filled, filled2, full_fill_kernel);
  filled2.forEach<double>([&filled](double& now, const int position[]) -> void {
    double d = filled.at<double>(position[0], position[1]);
    if (d > 0) {
      now = d;
    }
  });

  dst_grid.forEach<double>(
      [&filled2, &max_dist](double& now, const int position[]) -> void {
        double d = filled2.at<double>(position[0], position[1]);
        if (d > 0) {
          now = max_dist - d;
        }
      });
  */
}