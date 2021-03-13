#pragma once
#include <vector>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"

using namespace std;

void remove_noise(cv::Mat& src, cv::Mat& dst, cv::Mat& vs, EnvParams env_params,
                  double rad_coef = 0.01, int min_k = 2) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(
      new pcl::PointCloud<pcl::PointXYZ>);
  src.forEach<double>([&cloud_ptr, &env_params, &vs](
                          double& now, const int position[]) -> void {
    if (now <= 0) {
      return;
    }

    double x = now * (position[1] - env_params.width / 2) / env_params.f_xy;
    double y = now *
               (vs.at<int>(position[0], position[1]) - env_params.height / 2) /
               env_params.f_xy;
    cloud_ptr->points.push_back(pcl::PointXYZ(x, y, now));
  });

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud_ptr);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  cv::Mat full_grid =
      cv::Mat::zeros(env_params.height, env_params.width, CV_64FC1);
  for (int i = 0; i < cloud_ptr->points.size(); i++) {
    double x = cloud_ptr->points[i].x;
    double y = cloud_ptr->points[i].y;
    double z = cloud_ptr->points[i].z;
    double distance2 = x * x + y * y + z * z;

    //探索半径：係数*(距離)^2
    double radius = rad_coef * distance2;

    //最も近い点を探索し，半径r以内にあるか判定
    vector<int> pointIdxNKNSearch;
    vector<float> pointNKNSquaredDistance;
    int result = kdtree.radiusSearch((*cloud_ptr)[i], radius, pointIdxNKNSearch,
                                     pointNKNSquaredDistance, min_k);
    if (result == min_k) {
      int u = x / z * env_params.f_xy + env_params.width / 2;
      int v = y / z * env_params.f_xy + env_params.height / 2;
      full_grid.at<double>(v, u) = z;
    }
  }

  dst = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst.forEach<double>([&full_grid, &vs](double& now,
                                        const int position[]) -> void {
    now =
        full_grid.at<double>(vs.at<int>(position[0], position[1]), position[1]);
  });
}