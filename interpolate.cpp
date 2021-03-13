#pragma once
#include <chrono>
#include <map>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <time.h>
#include <opencv2/opencv.hpp>

#include "data/load_params.cpp"
#include "methods/ip_basic_cv.cpp"
/*
#include "methods/linear.cpp"
#include "methods/mrf.cpp"
#include "methods/original.cpp"
#include "methods/pwas.cpp"
*/
#include "models/env_params.cpp"
#include "models/hyper_params.cpp"
/*
#include "postprocess/generate_depth_image.cpp"
#include "preprocess/downsample.cpp"
#include "preprocess/find_neighbors.cpp"
#include "preprocess/grid_pcd.cpp"
*/
#include "postprocess/evaluate.cpp"
#include "postprocess/restore_pointcloud.cpp"
#include "preprocess/downsample.cpp"
#include "preprocess/grid_pointcloud.cpp"

using namespace std;

void interpolate(pcl::PointCloud<pcl::PointXYZ>& src_cloud, cv::Mat& img,
                 EnvParams env_params, HyperParams hyper_params,
                 string method_name, double& time, double& ssim, double& mse,
                 double& mre, double& f_val, bool show_cloud = false) {
  cv::Mat blured;
  cv::GaussianBlur(img, blured, cv::Size(5, 5), 1.0);

  auto start = chrono::system_clock::now();
  pcl::PointCloud<pcl::PointXYZ> downsampled;
  double PI = acos(-1);
  downsample(src_cloud, downsampled, -16.6, 16.6, 64, 16);

  cv::Mat grid, vs;
  grid_pointcloud(downsampled, -16.6, 16.6, 128, env_params, grid, vs);

  cv::Mat interpolated;

  if (method_name == "ip-basic") {
    ip_basic_cv(grid, interpolated, vs, env_params);
  }

  cv::Mat original_grid, original_vs;
  grid_pointcloud(src_cloud, -16.6, 16.6, 128, env_params, original_grid,
                  original_vs);

  time = chrono::duration_cast<chrono::milliseconds>(
             chrono::system_clock::now() - start)
             .count();
  evaluate(interpolated, original_grid, env_params, ssim, mse, mre, f_val);

  if (show_cloud) {
    pcl::PointCloud<pcl::PointXYZ> dst_cloud;
    restore_pointcloud(interpolated, vs, env_params, dst_cloud);
    pcl::visualization::CloudViewer viewer("Point Cloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(
        new pcl::PointCloud<pcl::PointXYZ>(dst_cloud));
    viewer.showCloud(cloud_ptr);
    while (!viewer.wasStopped()) {
    }
  }
}