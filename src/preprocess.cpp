#include <vector>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

#include "models.h"
#include "preprocess.h"

using namespace std;

/*
Downsample point cloud
*/
void downsample(pcl::PointCloud<pcl::PointXYZ>& src_cloud,
                pcl::PointCloud<pcl::PointXYZ>& dst_cloud,
                double min_angle_degree, double max_angle_degree,
                int original_layer_cnt, int down_layer_cnt) {
  double PI = acos(-1);
  double min_rad = min_angle_degree * PI / 180;
  double delta_rad = (max_angle_degree - min_angle_degree) /
                     (original_layer_cnt - 1) * PI / 180;

  dst_cloud = pcl::PointCloud<pcl::PointXYZ>();

  for (int i = 0; i < src_cloud.points.size(); i++) {
    double x = src_cloud.points[i].x;
    double y = src_cloud.points[i].y;
    double z = src_cloud.points[i].z;

    double r = sqrt(x * x + z * z);

    int idx = (int)((atan2(y, r) - min_rad) / delta_rad);
    if (idx < 0 || idx >= original_layer_cnt) {
      continue;
    }

    if (idx % (original_layer_cnt / down_layer_cnt) == 0) {
      dst_cloud.points.push_back(pcl::PointXYZ(x, y, z));
    }
  }
}

/*
Transform point cloud into depth image
カメラ座標系でLiDARグリッドを構築する
*/
void grid_pointcloud(const pcl::PointCloud<pcl::PointXYZ>& src_cloud,
                     double min_angle_degree, double max_angle_degree,
                     int target_layer_cnt, EnvParams& env_params, cv::Mat& grid,
                     cv::Mat& vs) {
  double PI = acos(-1);
  double min_rad = min_angle_degree * PI / 180;
  double delta_rad =
      (max_angle_degree - min_angle_degree) / (target_layer_cnt - 1) * PI / 180;

  // キャリブレーション
  grid = cv::Mat::zeros(target_layer_cnt, env_params.width, CV_64FC1);
  vs = cv::Mat::zeros(target_layer_cnt, env_params.width, CV_16SC1);

  double rollVal = (env_params.roll - 500) / 1000.0;
  double pitchVal = (env_params.pitch - 500) / 1000.0;
  double yawVal = (env_params.yaw - 500) / 1000.0;
  Eigen::MatrixXd calibration_mtx(4, 4);
  calibration_mtx << cos(yawVal) * cos(pitchVal),
      cos(yawVal) * sin(pitchVal) * sin(rollVal) - sin(yawVal) * cos(rollVal),
      cos(yawVal) * sin(pitchVal) * cos(rollVal) + sin(yawVal) * sin(rollVal),
      (env_params.X - 500) / 100.0, sin(yawVal) * cos(pitchVal),
      sin(yawVal) * sin(pitchVal) * sin(rollVal) + cos(yawVal) * cos(rollVal),
      sin(yawVal) * sin(pitchVal) * cos(rollVal) - cos(yawVal) * sin(rollVal),
      (env_params.Y - 500) / 100.0, -sin(pitchVal),
      cos(pitchVal) * sin(rollVal), cos(pitchVal) * cos(rollVal),
      (env_params.Z - 500) / 100.0, 0, 0, 0, 1;

  for (int i = 0; i < src_cloud.points.size(); i++) {
    double rawX = src_cloud.points[i].x;
    double rawY = src_cloud.points[i].y;
    double rawZ = src_cloud.points[i].z;

    double rawR = sqrt(rawX * rawX + rawZ * rawZ);
    double x = calibration_mtx(0, 0) * rawX + calibration_mtx(0, 1) * rawY +
               calibration_mtx(0, 2) * rawZ + calibration_mtx(0, 3);
    double y = calibration_mtx(1, 0) * rawX + calibration_mtx(1, 1) * rawY +
               calibration_mtx(1, 2) * rawZ + calibration_mtx(1, 3);
    double z = calibration_mtx(2, 0) * rawX + calibration_mtx(2, 1) * rawY +
               calibration_mtx(2, 2) * rawZ + calibration_mtx(2, 3);
    double r = sqrt(x * x + z * z);
    int v_idx = (int)((atan2(y, r) - min_rad) / delta_rad);

    if (z > 0) {
      int u = round(env_params.width / 2 + env_params.f_xy * x / z);
      int v = round(env_params.height / 2 + env_params.f_xy * y / z);
      if (0 <= u && u < env_params.width && 0 <= v && v < env_params.height &&
          0 <= v_idx && v_idx < vs.rows) {
        grid.at<double>(v_idx, u) = z;
        vs.at<ushort>(v_idx, u) = (ushort)v;
      }
    }
  }

  vs.forEach<ushort>([&](ushort& now, const int position[]) -> void {
    if (now > 0) {
      return;
    }

    int v = round(env_params.height / 2 +
                  env_params.f_xy * tan(min_rad + position[0] * delta_rad));
    if (0 <= v && v < env_params.height) {
      now = (ushort)v;
    }
  });
}

void remove_noise(cv::Mat& src, cv::Mat& dst, cv::Mat& vs, EnvParams env_params,
                  double rad_coef, int min_k) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 0; i < vs.rows; i++) {
    double* row = src.ptr<double>(i);
    for (int j = 0; j < vs.cols; j++) {
      double z = row[j];
      if (z <= 0) {
        continue;
      }

      double x = z * (j - env_params.width / 2) / env_params.f_xy;
      double y =
          z * (vs.at<ushort>(i, j) - env_params.height / 2) / env_params.f_xy;
      cloud_ptr->points.push_back(pcl::PointXYZ(x, y, z));
    }
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud_ptr);
  cv::Mat full_grid =
      cv::Mat::zeros(env_params.height, env_params.width, CV_64FC1);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
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
      int u = round(x / z * env_params.f_xy + env_params.width / 2);
      int v = round(y / z * env_params.f_xy + env_params.height / 2);
      if (0 <= u < env_params.width && 0 <= v && v < env_params.height) {
        full_grid.at<double>(v, u) = z;
        inliers->indices.push_back(i);
      }
    }
  }

  dst = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst.forEach<double>(
      [&full_grid, &vs](double& now, const int position[]) -> void {
        now = full_grid.at<double>(vs.at<ushort>(position[0], position[1]),
                                   position[1]);
      });
}