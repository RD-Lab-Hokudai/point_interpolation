#pragma once
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

#include "models.h"

using namespace std;

namespace qm {
// sigma on block_size
double sigma(cv::Mat& m, int i, int j, int block_size);

// Covariance
double cov(cv::Mat& m1, cv::Mat& m2, int i, int j, int block_size);

// Mean reprojection error
double mre(cv::Mat& img1, cv::Mat& img2);

// Mean squared error
double eqm(cv::Mat& img1, cv::Mat& img2);

// Compute the PSNR between 2 images
double psnr(cv::Mat& img_src, cv::Mat& img_compressed, int block_size);

// Compute the SSIM between 2 images
double ssim(cv::Mat& img1, cv::Mat& img2, int block_size);

// Compute the f value between img1 (original) and img2 (reference)
double f_value(cv::Mat& img1, cv::Mat& img2);
}  // namespace qm

void evaluate(cv::Mat& grid, cv::Mat& original_grid, EnvParams& env_params,
              double& ssim, double& mse, double& mre, double& f_val);

void restore_pointcloud(cv::Mat& grid, cv::Mat& vs, EnvParams env_params,
                        pcl::PointCloud<pcl::PointXYZ>& dst_cloud);

void generate_depth_image(cv::Mat &grid, cv::Mat &img);