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
                 double& mre, bool show_cloud = false) {
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
  grid_pointcloud(src_cloud, -16.6, 16.6, env_params.height, env_params,
                  original_grid, original_vs);

  double f_val;
  // evaluate(interpolated, original_grid, env_params, ssim, mse, mre, f_val);

  if (show_cloud) {
    cv::imshow("I", interpolated);
    cv::waitKey();
    pcl::PointCloud<pcl::PointXYZ> dst_cloud;
    restore_pointcloud(interpolated, vs, env_params, dst_cloud);
    pcl::visualization::CloudViewer viewer("Point Cloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(
        new pcl::PointCloud<pcl::PointXYZ>(dst_cloud));
    viewer.showCloud(cloud_ptr);
    while (!viewer.wasStopped()) {
    }
  }

  /*
    vector<vector<double>> original_grid, filtered_grid,
        original_interpolate_grid, filtered_interpolate_grid;
    vector<vector<int>> target_vs, base_vs;
    int layer_cnt = 16;
    grid_pcd(pcd_ptr, envParams, original_grid, filtered_grid,
             original_interpolate_grid, filtered_interpolate_grid, target_vs,
             base_vs, layer_cnt);

    vector<vector<int>> original_vs;
    if (envParams.isFullHeight) {
      original_vs =
          vector<vector<int>>(envParams.height, vector<int>(envParams.width,
    0)); for (int i = 0; i < envParams.height; i++) { for (int j = 0; j <
    envParams.width; j++) { original_vs[i][j] = i;
        }
      }
      swap(original_vs, target_vs);
      cout << original_vs.size() << endl;
      cout << target_vs.size() << endl;
    } else {
      original_vs = target_vs;
    }
    */

  /*
  { // snow removal
      vector<vector<double>> removed_grid;
      auto start_tmp = chrono::system_clock::now();
      remove_noise(filtered_grid, removed_grid, base_vs, envParams, 0.0015);
      cout <<
  chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() -
  start_tmp).count() << "ms" << endl;

      shared_ptr<geometry::PointCloud> filtered_ptr, removed_ptr, removed_ptr2;
      restore_pcd_simple(filtered_grid, base_vs, envParams, filtered_ptr);
      restore_pcd_simple(removed_grid, base_vs, envParams, removed_ptr, 100);
      cout << filtered_ptr->points_.size() << " " << removed_ptr->points_.size()
  << endl; visualization::DrawGeometries({filtered_ptr, removed_ptr});
  }
  */

  {
    // vector<vector<vector<int>>> neighbors;
    // find_neighbors(envParams, original_grid, original_vs, neighbors, 30);
  }

  /*
    vector<vector<double>> interpolated_z;
    if (envParams.method == "linear") {
      linear(interpolated_z, filtered_interpolate_grid, target_vs, base_vs,
             envParams);
    }
    if (envParams.method == "mrf") {
      mrf(interpolated_z, filtered_grid, filtered_interpolate_grid, target_vs,
          base_vs, envParams, blured, hyperParams.mrf_k, hyperParams.mrf_c);
    }
    if (envParams.method == "ip-basic") {
      ip_basic(interpolated_z, filtered_grid, target_vs, base_vs, envParams);
    }
    if (envParams.method == "pwas") {
      pwas(interpolated_z, filtered_grid, target_vs, base_vs, envParams, blured,
           hyperParams.pwas_sigma_c, hyperParams.pwas_sigma_s,
           hyperParams.pwas_sigma_r, hyperParams.pwas_r);
    }
    if (envParams.method == "original") {
      original(interpolated_z, filtered_grid, target_vs, base_vs, envParams,
               blured, hyperParams.original_color_segment_k,
               hyperParams.original_sigma_s, hyperParams.original_sigma_r,
               hyperParams.original_r, hyperParams.original_coef_s);
    }
    cout << "Done" << endl;
    */

  /*
  {
      cout << interpolated_z.size() << endl;
      double max_original_depth = 0;
      double min_original_depth = 100;
      for (int i = 0; i < filtered_grid.size(); i++)
      {
          for (int j = 0; j < envParams.width; j++)
          {
              if (filtered_grid[i][j] < 100)
              {
                  max_original_depth = max(max_original_depth,
  filtered_grid[i][j]);
              }
              min_original_depth = min(min_original_depth, filtered_grid[i][j]);
          }
      }

      double max_depth = 0;
      double min_depth = 100;
      for (int i = 0; i < interpolated_z.size(); i++)
      {
          for (int j = 0; j < envParams.width; j++)
          {
              if (interpolated_z[i][j] < 100)
              {
                  max_depth = max(max_depth, interpolated_z[i][j]);
              }
              min_depth = min(min_depth, interpolated_z[i][j]);
          }
      }
      cv::Mat original_img = cv::Mat::zeros(64 * 2, envParams.width * 2,
  CV_8UC1); for (int i = 0; i < filtered_grid.size(); i++)
      {
          for (int j = 0; j < envParams.width; j++)
          {
              uchar val = 255 * (filtered_grid[i][j] - min_original_depth) /
  (max_original_depth - min_original_depth); original_img.at<uchar>(i * 4 * 2, j
  * 2) = val; original_img.at<uchar>(i * 4 * 2 + 1, j * 2) = val;
              original_img.at<uchar>(i * 4 * 2, j * 2 + 1) = val;
              original_img.at<uchar>(i * 4 * 2 + 1, j * 2 + 1) = val;
          }
      }
      cv::Mat img = cv::Mat::zeros(64 * 2, envParams.width * 2, CV_8UC1);
      for (int i = 0; i < interpolated_z.size(); i++)
      {
          for (int j = 0; j < envParams.width; j++)
          {
              uchar val = 255 * (interpolated_z[i][j] - min_depth) / (max_depth
  - min_depth); img.at<uchar>(i * 2, j * 2) = val; img.at<uchar>(i * 2 + 1, j *
  2) = val; img.at<uchar>(i * 2, j * 2 + 1) = val; img.at<uchar>(i * 2 + 1, j *
  2 + 1) = val;
          }
      }
      cout << "en" << endl;
      cv::Mat original_cm_img, cm_img;
      cv::applyColorMap(original_img, original_cm_img, COLORMAP_JET);
      cv::applyColorMap(img, cm_img, COLORMAP_JET);
      cv::imshow("A", original_cm_img);
      cv::imshow("B", cm_img);
      cv::waitKey();
  }
  */

  /*
  {
  cv::Mat grid_img = cv::Mat::zeros(target_vs.size(), envParams.width, CV_8UC3);
  auto filtered_ptr = make_shared<geometry::PointCloud>();
      for (int i = 0; i < original_vs.size(); i++)
      {
          for (int j = 0; j < envParams.width; j++)
          {
              if (original_grid[i][j] > 0)
              {
                  grid_img.at<cv::Vec3b>(i, j) = cv::Vec3b(100 * filtered_grid[i
  / 4][j], 100 * filtered_grid[i / 4][j], 0); double z = original_grid[i][j];
                  double x = z * (j - envParams.width / 2) / envParams.f_xy;
                  double y = z * (target_vs[i][j] - envParams.height / 2) /
  envParams.f_xy; if (i % (64 / layer_cnt) == 0)
                  {
                      filtered_ptr->points_.emplace_back(x, y, z);
                  }
              }
          }
      }

      //cv::imshow("aa", grid_img);
      //cv::waitKey();
      //visualization::DrawGeometries({filtered_ptr});
  }
  */

  /*
    { // Evaluate
      time = chrono::duration_cast<chrono::milliseconds>(
                 chrono::system_clock::now() - start)
                 .count();
      double f_val;
      evaluate(interpolated_z, original_grid, target_vs, original_vs, envParams,
               layer_cnt, ssim, mse, mre, f_val);
      if (show_result) {
        cout << time << "ms" << endl;
        cout << "SSIM = " << fixed << setprecision(5) << ssim << endl;
        cout << "MSE = " << mse << endl;
        cout << "MRE = " << mre << endl;
        cout << "F value = " << f_val << endl;
      }
    }
  */

  /*
    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    auto original_ptr = make_shared<geometry::PointCloud>();
    restore_pcd(interpolated_z, original_grid, target_vs, original_vs,
    envParams, blured, interpolated_ptr, original_ptr);

    if (show_pcd) {
      visualization::DrawGeometries({pcd_ptr}, "Raw");
      generate_depth_image(interpolated_z, target_vs, envParams);
      visualization::DrawGeometries({original_ptr}, "Downsampled", 1000, 800);
      visualization::DrawGeometries({interpolated_ptr}, "Interpolated", 1000,
                                    800);
    }

    if (!io::WritePointCloudToPCD(envParams.folder_path + to_string(data_no) +
                                      "_interpolated.pcd",
                                  *interpolated_ptr, {true})) {
      cout << "Cannot write" << endl;
    }
    if (!io::WritePointCloudToPCD(envParams.folder_path + to_string(data_no) +
                                      "_original.pcd",
                                  *original_ptr, {true})) {
      cout << "Cannot write" << endl;
    }
    */
}