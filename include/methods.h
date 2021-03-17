#pragma once
#include <opencv2/opencv.hpp>

#include "models.h"

using namespace std;

void linear(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
            EnvParams env_params);

void ip_basic(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
              EnvParams env_params);

void mrf(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
         EnvParams env_params, cv::Mat img, double k, double c);

void guided_filter(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
                   EnvParams env_params, cv::Mat img);

void pwas(const cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs, cv::Mat& img,
          double sigma_c, double sigma_s, double sigma_r, double r);

void original(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
              EnvParams& env_params, cv::Mat& img, double color_segment_k,
              double sigma_s, int r, double coef_s);