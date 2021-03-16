#pragma once
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"
#include "./linear.cpp"

using namespace std;

void mrf(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
         EnvParams env_params, cv::Mat img, double k, double c) {
  cv::Mat linear_grid;
  linear(src_grid, linear_grid, vs, env_params);

  int length = vs.rows * vs.cols;
  Eigen::VectorXd z_line(length);
  Eigen::SparseMatrix<double> W(length, length);
  vector<Eigen::Triplet<double>> W_triplets;
  for (int i = 0; i < vs.rows; i++) {
    for (int j = 0; j < vs.cols; j++) {
      double now = src_grid.at<double>(i, j);
      if (now > 0) {
        W_triplets.emplace_back(i * vs.cols + j, i * vs.cols + j, k);
      }
      z_line[i * vs.cols + j] = linear_grid.at<double>(i, j);
    }
  }
  W.setFromTriplets(W_triplets.begin(), W_triplets.end());
  cout << "A" << endl;

  Eigen::SparseMatrix<double> S(length, length);
  vector<Eigen::Triplet<double>> S_triplets;
  int dires = 4;
  int dx[4] = {1, -1, 0, 0};
  int dy[4] = {0, 0, 1, -1};
  for (int i = 0; i < vs.rows; i++) {
    for (int j = 0; j < vs.cols; j++) {
      double wSum = 0;
      ushort v0 = vs.at<ushort>(i, j);
      for (int k = 0; k < dires; k++) {
        int x = j + dx[k];
        int y = i + dy[k];
        if (0 <= x && x < vs.cols && 0 <= y && y < vs.rows) {
          int v1 = vs.at<double>(y, x);
          double x_norm2 =
              cv::norm(img.at<cv::Vec3b>(v0, j) - img.at<cv::Vec3b>(v1, x)) /
              (255 * 255);
          double w = -sqrt(exp(-c * x_norm2));
          S_triplets.emplace_back(i * vs.cols + j, y * vs.cols + x, w);
          wSum += w;
        }
      }
      S_triplets.emplace_back(i * vs.cols + j, i * vs.cols + j, -wSum);
    }
  }
  S.setFromTriplets(S_triplets.begin(), S_triplets.end());
  Eigen::SparseMatrix<double> A = S.transpose() * S + W.transpose() * W;
  Eigen::VectorXd b = W.transpose() * W * z_line;
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                           Eigen::Lower | Eigen::Upper>
      cg;
  cg.compute(A);
  Eigen::VectorXd y_res = cg.solve(b);
  cout << "B" << endl;

  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst_grid.forEach<double>(
      [&vs, &y_res](double& now, const int position[]) -> void {
        now = y_res[position[0] * vs.cols + position[1]];
      });
}