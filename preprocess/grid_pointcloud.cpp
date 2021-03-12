#pragma once
#include <vector>

#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"

using namespace std;

/*
Transform point cloud into depth image
*/
void grid_pointcloud(pcl::PointCloud<pcl::PointXYZ> &src_cloud,
                     double min_angle_degree, double max_angle_degree,
                     int target_layer_cnt, EnvParams &env_params, cv::Mat &grid,
                     cv::Mat &vs)
{
  double PI = acos(-1);
  double min_rad = min_angle_degree * PI / 180;
  double delta_rad =
      (max_angle_degree - min_angle_degree) / (target_layer_cnt - 1) * PI / 180;

  grid = cv::Mat::zeros(target_layer_cnt, env_params.width, CV_64FC1);
  vs = cv::Mat::zeros(target_layer_cnt, env_params.width, CV_16SC1);
  cv::Mat vs2 = cv::Mat::zeros(target_layer_cnt, env_params.width, CV_64FC1);

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

  for (int i = 0; i < src_cloud.points.size(); i++)
  {
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

    int v_idx = (int)((atan2(rawY, rawR) - min_rad) / delta_rad);

    if (z > 0)
    {
      int u = (int)(env_params.width / 2 + env_params.f_xy * x / z);
      int v = (int)(env_params.height / 2 + env_params.f_xy * y / z);
      if (0 <= u && u < env_params.width && 0 <= v && v < env_params.height)
      {
        grid.at<double>(v_idx, u) = z;
        //vs.at<ushort>(v_idx, u) = (ushort)v;
        vs2.at<double>(v_idx, u) = v / 300.0;
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZ> tst_cloud;
  cv::Mat hoge = cv::Mat::zeros(env_params.height, env_params.width, CV_8UC1);
  for (int i = 0; i < vs.rows; i++)
  {
    for (int j = 0; j < vs.cols; j++)
    {
      double rawY = tan(min_rad + i * delta_rad);
      double A = env_params.f_xy * calibration_mtx(0, 0) +
                 (env_params.width / 2 - j) * calibration_mtx(2, 0);
      double B = env_params.f_xy * calibration_mtx(0, 1) +
                 (env_params.width / 2 - j) * calibration_mtx(2, 1);
      double C = env_params.f_xy * calibration_mtx(0, 2) +
                 (env_params.width / 2 - j) * calibration_mtx(2, 2);
      double D = env_params.f_xy * calibration_mtx(0, 3) +
                 (env_params.width / 2 - j) * calibration_mtx(2, 3);
      double E = B * rawY + D;

      // 判別式
      if (A * A + C * C - E * E < 0)
      {
        return;
      }

      double rawX1 = (-A * E - C * sqrt(A * A + C * C - E * E)) / (A * A + C * C);
      double rawX2 = (-A * E + C * sqrt(A * A + C * C - E * E)) / (A * A + C * C);
      double rawX = 0;
      double rawZ = 1;
      // Z >= 0は前提とする
      if ((A * rawX1 + E) / C <= 0)
      {
        rawX = rawX1;
      }
      else
      {
        // このケースはほぼ無し
        rawX = rawX2;
      }
      rawZ = sqrt(1 - rawX * rawX);

      double x = calibration_mtx(0, 0) * rawX + calibration_mtx(0, 1) * rawY +
                 calibration_mtx(0, 2) * rawZ + calibration_mtx(0, 3);
      double y = calibration_mtx(1, 0) * rawX + calibration_mtx(1, 1) * rawY +
                 calibration_mtx(1, 2) * rawZ + calibration_mtx(1, 3);
      double z = calibration_mtx(2, 0) * rawX + calibration_mtx(2, 1) * rawY +
                 calibration_mtx(2, 2) * rawZ + calibration_mtx(2, 3);

      tst_cloud.points.push_back(pcl::PointXYZ(x, y, z));

      int u = (int)(env_params.width / 2 + env_params.f_xy * x / z);
      if (abs(u - j) > 1)
      {
        cout << i << " " << j << " : " << u << endl;
      }
      int v = (int)(env_params.height / 2 + env_params.f_xy * y / z);
      if (0 <= u && u < env_params.width && 0 <= v && v < env_params.height)
      {
        hoge.at<int>(v, u) = i * 10;
      }
    }
  }
  cv::imshow("hoge", hoge);
  cv::waitKey();
  {
    pcl::visualization::CloudViewer viewer("Point Cloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(
        new pcl::PointCloud<pcl::PointXYZ>(tst_cloud));
    viewer.showCloud(cloud_ptr);
    while (!viewer.wasStopped())
    {
    }
  }

  vs.forEach<ushort>([&](ushort &now, const int position[]) -> void {
    if (now > 0)
    {
      return;
    }

    /* Old strategy
        double rawZ = 1;
        double rawY = rawZ * tan(min_rad + position[0] * delta_rad);
        double x_coef =
            env_params.f_xy * calibration_mtx(0, 0) -
            (position[1] - env_params.width / 2) * calibration_mtx(2, 0);
        double right_value =
            ((position[1] - env_params.width / 2) * calibration_mtx(2, 1) -
             env_params.f_xy * calibration_mtx(0, 1)) *
                rawY +
            ((position[1] - env_params.width / 2) * calibration_mtx(2, 2) -
             env_params.f_xy * calibration_mtx(0, 2)) *
                rawZ -
            env_params.f_xy * calibration_mtx(0, 3) +
            (position[1] - env_params.width / 2) * calibration_mtx(2, 3);
        double rawX = right_value / x_coef;

        double y = calibration_mtx(1, 0) * rawX + calibration_mtx(1, 1) * rawY +
                   calibration_mtx(1, 2) * rawZ;  //+ calibration_mtx(1, 3);
        double z = calibration_mtx(2, 0) * rawX + calibration_mtx(2, 1) * rawY +
                   calibration_mtx(2, 2) * rawZ;  // + calibration_mtx(2, 3);
        now = (ushort)(env_params.height / 2 + env_params.f_xy * y / z);
        vs2.at<double>(position[0], position[1]) = now / 300.0;
        */

    /* Second strategy
    double rawY = tan(min_rad + position[0] * delta_rad);
    double A = env_params.f_xy * calibration_mtx(0, 0) +
               (env_params.width / 2 - position[1]) * calibration_mtx(2, 0);
    double B = (env_params.f_xy * calibration_mtx(0, 1) +
                (env_params.width / 2 - position[1]) * calibration_mtx(2, 1)) *
               rawY;
    double C = env_params.f_xy * calibration_mtx(0, 2) +
               (env_params.width / 2 - position[1]) * calibration_mtx(2, 2);
    double D = env_params.f_xy * calibration_mtx(0, 3) +
               (env_params.width / 2 - position[1]) * calibration_mtx(2, 3);
    double E = B + D;

    double rawZ = (-C * E + A * sqrt(A * A + C * C - E * E)) / (A * A + C * C);
    double rawX = sqrt(1 - rawZ * rawZ);
    double x = calibration_mtx(0, 0) * rawX + calibration_mtx(0, 1) * rawY +
               calibration_mtx(0, 2) * rawZ + calibration_mtx(0, 3);
    double y = calibration_mtx(1, 0) * rawX + calibration_mtx(1, 1) * rawY +
               calibration_mtx(1, 2) * rawZ + calibration_mtx(1, 3);
    double z = calibration_mtx(2, 0) * rawX + calibration_mtx(2, 1) * rawY +
               calibration_mtx(2, 2) * rawZ + calibration_mtx(2, 3);
    ushort v = (ushort)(env_params.height / 2 + env_params.f_xy * y / z);
    if (0 <= v && v < env_params.height) {
      now = v;
      int uu = (int)(env_params.width / 2 + env_params.f_xy * x / z);
      vs2.at<double>(position[0], position[1]) = now / 300.0;
    }
    */

    double rawY = tan(min_rad + position[0] * delta_rad);
    double A = env_params.f_xy * calibration_mtx(0, 0) +
               (env_params.width / 2 - position[1]) * calibration_mtx(2, 0);
    double B = env_params.f_xy * calibration_mtx(0, 1) +
               (env_params.width / 2 - position[1]) * calibration_mtx(2, 1);
    double C = env_params.f_xy * calibration_mtx(0, 2) +
               (env_params.width / 2 - position[1]) * calibration_mtx(2, 2);
    double D = env_params.f_xy * calibration_mtx(0, 3) +
               (env_params.width / 2 - position[1]) * calibration_mtx(2, 3);
    double E = B * rawY + D;

    // 判別式
    if (A * A + C * C - E * E < 0)
    {
      return;
    }

    double rawX1 = (-A * E - C * sqrt(A * A + C * C - E * E)) / (A * A + C * C);
    double rawX2 = (-A * E + C * sqrt(A * A + C * C - E * E)) / (A * A + C * C);
    double rawX = 0;
    double rawZ = 1;
    // Z >= 0は前提とする
    if ((A * rawX1 + E) / C <= 0)
    {
      rawX = rawX1;
    }
    else
    {
      rawX = rawX2;
    }
    rawZ = sqrt(1 - rawX * rawX);

    double x = calibration_mtx(0, 0) * rawX + calibration_mtx(0, 1) * rawY +
               calibration_mtx(0, 2) * rawZ + calibration_mtx(0, 3);
    double y = calibration_mtx(1, 0) * rawX + calibration_mtx(1, 1) * rawY +
               calibration_mtx(1, 2) * rawZ + calibration_mtx(1, 3);
    double z = calibration_mtx(2, 0) * rawX + calibration_mtx(2, 1) * rawY +
               calibration_mtx(2, 2) * rawZ + calibration_mtx(2, 3);
    int v = (int)(env_params.height / 2 + env_params.f_xy * y / z);

    double s = calibration_mtx(2, 0) * rawX + calibration_mtx(2, 1) * rawY + calibration_mtx(2, 2) * rawZ + calibration_mtx(2, 3);
    v = (int)(((env_params.f_xy * calibration_mtx(1, 0) +
                env_params.height / 2 * calibration_mtx(2, 0)) *
                   rawX +
               (env_params.f_xy * calibration_mtx(1, 1) +
                env_params.height / 2 * calibration_mtx(2, 1)) *
                   rawY +
               (env_params.f_xy * calibration_mtx(1, 2) +
                env_params.height / 2 * calibration_mtx(2, 2)) *
                   rawZ +
               (env_params.f_xy * calibration_mtx(1, 3) +
                env_params.height / 2 * calibration_mtx(2, 3))) /
              s);
    if (0 <= v && v < env_params.height)
    {
      now = (ushort)v;
      int uu = (int)(env_params.width / 2 + env_params.f_xy * x / z);
      vs2.at<double>(position[0], position[1]) = v / 300.0;
    }
  });

  cv::imshow(to_string(target_layer_cnt), grid);
  cv::imshow("A", hoge);
  cv::waitKey();
  cv::imshow("B", vs2);
  cv::waitKey();
}
