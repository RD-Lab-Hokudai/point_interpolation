#pragma once
#include <vector>

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>

#include "../models/envParams.cpp"
#include "remove_noise.cpp"

using namespace std;
using namespace open3d;

/*
Transform point cloud into depth image
*/
void grid_pcd(shared_ptr<geometry::PointCloud> raw_pcd_ptr, EnvParams envParams,
              vector<vector<double>> &original_grid,
              vector<vector<double>> &filtered_grid,
              vector<vector<double>> &original_interpolate_grid,
              vector<vector<double>> &filtered_interpolate_grid,
              vector<vector<int>> &target_vs, vector<vector<int>> &base_vs,
              int layer_cnt = 16) {
  vector<double> tans;
  double PI = acos(-1);
  double delta_rad = 0.52698 * PI / 180;
  double max_rad = (16.6 + 0.26349) * PI / 180;
  double rad = (-16.6 + 0.26349) * PI / 180;
  while (rad < max_rad + 0.00001) {
    tans.emplace_back(tan(rad));
    rad += delta_rad;
  }

  auto clipped_ptr = make_shared<geometry::PointCloud>();
  vector<int> clipped_indecies;
  vector<vector<Eigen::Vector3d>> all_layers(64, vector<Eigen::Vector3d>());
  double rollVal = (envParams.roll - 500) / 1000.0;
  double pitchVal = (envParams.pitch - 500) / 1000.0;
  double yawVal = (envParams.yaw - 500) / 1000.0;
  Eigen::MatrixXd calibration_mtx(4, 4);
  calibration_mtx << cos(yawVal) * cos(pitchVal),
      cos(yawVal) * sin(pitchVal) * sin(rollVal) - sin(yawVal) * cos(rollVal),
      cos(yawVal) * sin(pitchVal) * cos(rollVal) + sin(yawVal) * sin(rollVal),
      (envParams.X - 500) / 100.0, sin(yawVal) * cos(pitchVal),
      sin(yawVal) * sin(pitchVal) * sin(rollVal) + cos(yawVal) * cos(rollVal),
      sin(yawVal) * sin(pitchVal) * cos(rollVal) - cos(yawVal) * sin(rollVal),
      (envParams.Y - 500) / 100.0, -sin(pitchVal), cos(pitchVal) * sin(rollVal),
      cos(pitchVal) * cos(rollVal), (envParams.Z - 500) / 100.0, 0, 0, 0, 1;
  cout << calibration_mtx << endl;

  for (int i = 0; i < raw_pcd_ptr->points_.size(); i++) {
    double rawX = raw_pcd_ptr->points_[i][1];
    double rawY = -raw_pcd_ptr->points_[i][2];
    double rawZ = -raw_pcd_ptr->points_[i][0];

    double r = sqrt(rawX * rawX + rawZ * rawZ);
    double x = calibration_mtx(0, 0) * rawX + calibration_mtx(0, 1) * rawY +
               calibration_mtx(0, 2) * rawZ + calibration_mtx(0, 3);
    double y = calibration_mtx(1, 0) * rawX + calibration_mtx(1, 1) * rawY +
               calibration_mtx(1, 2) * rawZ + calibration_mtx(1, 3);
    double z = calibration_mtx(2, 0) * rawX + calibration_mtx(2, 1) * rawY +
               calibration_mtx(2, 2) * rawZ + calibration_mtx(2, 3);

    if (z > 0) {
      int u = (int)(envParams.width / 2 + envParams.f_xy * x / z);
      int v = (int)(envParams.height / 2 + envParams.f_xy * y / z);
      if (0 <= u && u < envParams.width && 0 <= v && v < envParams.height) {
        auto it = lower_bound(tans.begin(), tans.end(), rawY / r);
        int index = it - tans.begin();
        all_layers[index].emplace_back(x, y, z);
        clipped_ptr->points_.emplace_back(x, y, z);
        clipped_indecies.emplace_back(index);
      }
    }
  }

  /*
  {
      auto start = chrono::system_clock::now();
      all_layers = remove_snow(clipped_ptr, all_layers, clipped_indecies);
      cout <<
  chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() -
  start).count() << "ms" << endl;
  }
  */

  for (int i = 0; i < 64; i++) {
    // Remove occlusion
    // no sort
    vector<Eigen::Vector3d> removed;
    for (size_t j = 0; j < all_layers[i].size(); j++) {
      while (removed.size() > 0 &&
             removed.back()[0] * all_layers[i][j][2] >=
                 all_layers[i][j][0] * removed.back()[2]) {
        removed.pop_back();
      }
      removed.emplace_back(all_layers[i][j]);
    }
  }

  target_vs = vector<vector<int>>(64, vector<int>(envParams.width, 0));
  base_vs = vector<vector<int>>(layer_cnt, vector<int>(envParams.width, 0));
  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < envParams.width; j++) {
      double tan = tans[i];
      double rawZ = 1;
      double rawY = rawZ * tan;
      double x_coef = envParams.f_xy * calibration_mtx(0, 0) -
                      (j - envParams.width / 2) * calibration_mtx(2, 0);
      double right_value = ((j - envParams.width / 2) * calibration_mtx(2, 1) -
                            envParams.f_xy * calibration_mtx(0, 1)) *
                               rawY +
                           ((j - envParams.width / 2) * calibration_mtx(2, 2) -
                            envParams.f_xy * calibration_mtx(0, 2)) *
                               rawZ;
      double rawX = right_value / x_coef;
      double y = calibration_mtx(1, 0) * rawX + calibration_mtx(1, 1) * rawY +
                 calibration_mtx(1, 2) * rawZ;
      double z = calibration_mtx(2, 0) * rawX + calibration_mtx(2, 1) * rawY +
                 calibration_mtx(2, 2) * rawZ;
      int v = (int)(envParams.f_xy * y + envParams.height / 2 * z) / z;

      target_vs[i][j] = max(v, 0);
    }
  }

  original_grid =
      vector<vector<double>>(64, vector<double>(envParams.width, -1));
  filtered_grid =
      vector<vector<double>>(layer_cnt, vector<double>(envParams.width, -1));
  original_interpolate_grid =
      vector<vector<double>>(64, vector<double>(envParams.width, -1));
  filtered_interpolate_grid =
      vector<vector<double>>(layer_cnt, vector<double>(envParams.width, -1));
  for (int i = 0; i < 64; i++) {
    if (all_layers[i].size() == 0) {
      continue;
    }

    int now = 0;
    int uPrev =
        (int)(envParams.width / 2 +
              envParams.f_xy * all_layers[i][0][0] / all_layers[i][0][2]);
    int vPrev =
        (int)(envParams.height / 2 +
              envParams.f_xy * all_layers[i][0][1] / all_layers[i][0][2]);
    while (now < uPrev) {
      original_interpolate_grid[i][now] = all_layers[i][0][2];
      now++;
    }
    for (int j = 0; j + 1 < all_layers[i].size(); j++) {
      int u =
          (int)(envParams.width / 2 + envParams.f_xy * all_layers[i][j + 1][0] /
                                          all_layers[i][j + 1][2]);
      int v = (int)(envParams.height / 2 + envParams.f_xy *
                                               all_layers[i][j + 1][1] /
                                               all_layers[i][j + 1][2]);
      original_grid[i][uPrev] = all_layers[i][j][2];
      target_vs[i][uPrev] = vPrev;

      while (now < min(envParams.width, u)) {
        double angle = (all_layers[i][j + 1][2] - all_layers[i][j][2]) /
                       (all_layers[i][j + 1][0] - all_layers[i][j][0]);
        double tan = (now - envParams.width / 2) / envParams.f_xy;
        double z = (all_layers[i][j][2] - angle * all_layers[i][j][0]) /
                   (1 - tan * angle);
        original_interpolate_grid[i][now] = z;
        now++;
      }
      uPrev = u;
      vPrev = v;
    }

    original_grid[i][uPrev] = all_layers[i].back()[2];
    target_vs[i][uPrev] = vPrev;
    while (now < envParams.width) {
      original_interpolate_grid[i][now] = all_layers[i].back()[2];
      now++;
    }
  }
  for (int i = 0; i < layer_cnt; i++) {
    for (int j = 0; j < envParams.width; j++) {
      filtered_grid[i][j] = original_grid[i * (64 / layer_cnt)][j];
      filtered_interpolate_grid[i][j] =
          original_interpolate_grid[i * (64 / layer_cnt)][j];
      base_vs[i][j] = target_vs[i * (64 / layer_cnt)][j];
    }
  }

  /*
  { // Check
      auto original_ptr = make_shared<geometry::PointCloud>();
      auto filtered_ptr = make_shared<geometry::PointCloud>();
      for (int i = 0; i < 64; i++)
      {
          for (int j = 0; j < envParams.width; j++)
          {
              double z = original_grid[i][j];
              if (z < 0)
              {
                  continue;
              }
              double x = z * (j - envParams.width / 2) / envParams.f_xy;
              double y = z * (target_vs[i][j] - envParams.height / 2) /
  envParams.f_xy; original_ptr->points_.emplace_back(x, y, z);
          }

          if (i % (64 / layer_cnt) == 0)
          {
              for (int j = 0; j < envParams.width; j++)
              {
                  double z = filtered_grid[i / (64 / layer_cnt)][j];
                  if (z < 0)
                  {
                      continue;
                  }
                  double x = z * (j - envParams.width / 2) / envParams.f_xy;
                  double y = z * (target_vs[i][j] - envParams.height / 2) /
  envParams.f_xy; filtered_ptr->points_.emplace_back(x, z, -y);
              }
          }
      }
      //visualization::DrawGeometries({filtered_ptr}, "Points", 1200, 720);
  }
  */
}