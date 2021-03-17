#pragma once
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

#include "models.h"

using namespace std;

/*
Downsample point cloud
*/
void downsample(pcl::PointCloud<pcl::PointXYZ>& src_cloud,
                pcl::PointCloud<pcl::PointXYZ>& dst_cloud,
                double min_angle_degree, double max_angle_degree,
                int original_layer_cnt, int down_layer_cnt);

/*
Transform point cloud into depth image
カメラ座標系でLiDARグリッドを構築する
*/
void grid_pointcloud(const pcl::PointCloud<pcl::PointXYZ>& src_cloud,
                     double min_angle_degree, double max_angle_degree,
                     int target_layer_cnt, EnvParams& env_params, cv::Mat& grid,
                     cv::Mat& vs);

void remove_noise(cv::Mat& src, cv::Mat& dst, cv::Mat& vs, EnvParams env_params,
                  double rad_coef = 0.01, int min_k = 2);