#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"
#include "quality_metrics.cpp"

using namespace std;

void evaluate(cv::Mat& grid, cv::Mat& original_grid, EnvParams& env_params,
              double& ssim, double& mse, double& mre, double& f_val) {
  ssim = qm::ssim(original_grid, grid, 4);
  mse = qm::eqm(original_grid, grid);
  mre = qm::mre(original_grid, grid);
  f_val = qm::f_value(original_grid, grid);
}