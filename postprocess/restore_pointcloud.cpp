#pragma once
#include <vector>

#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#include "../models/env_params.cpp"

using namespace std;

void restore_pointcloud(cv::Mat &grid, cv::Mat &vs, EnvParams env_params,
                        pcl::PointCloud<pcl::PointXYZ> &dst_cloud)
{
  dst_cloud = pcl::PointCloud<pcl::PointXYZ>();

  for (int i = 0; i < vs.rows; i++)
  {
    double *row = grid.ptr<double>(i);
    for (int j = 0; j < vs.cols; j++)
    {
      double z = grid.at<double>(i, j);
      // row[j];
      if (z <= 0)
      {
        //continue;
      }

      double x = z * (j - env_params.width / 2) / env_params.f_xy;
      double y =
          z * (vs.at<ushort>(i, j) - env_params.height / 2) / env_params.f_xy;
      dst_cloud.points.push_back(pcl::PointXYZ(x, y, z));
    }
  }
}