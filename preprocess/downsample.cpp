#pragma once
#include <vector>

#include <pcl/point_cloud.h>

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