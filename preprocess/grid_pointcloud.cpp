#pragma once
#include <vector>

#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"

using namespace std;

/*
Transform point cloud into depth image
カメラ座標系でLiDARグリッドを構築する
*/
void grid_pointcloud(pcl::PointCloud<pcl::PointXYZ>& src_cloud,
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
      int u = (int)(env_params.width / 2 + env_params.f_xy * x / z);
      int v = (int)(env_params.height / 2 + env_params.f_xy * y / z);
      if (0 <= u && u < env_params.width && 0 <= v && v < env_params.height) {
        grid.at<double>(v_idx, u) = z;
        vs.at<ushort>(v_idx, u) = (ushort)v;
      }
    }
  }

  vs.forEach<ushort>([&](ushort& now, const int position[]) -> void {
    if (now > 0) {
      return;
    }

    int v = (int)(env_params.height / 2 +
                  env_params.f_xy * tan(min_rad + position[0] * delta_rad));
    if (0 <= v && v < env_params.height) {
      now = v;
    }
  });
}
