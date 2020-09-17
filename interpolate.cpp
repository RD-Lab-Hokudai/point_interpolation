#pragma once
#include <vector>
#include <chrono>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <time.h>

#include "models/envParams.cpp"
#include "models/hyperParams.cpp"
#include "data/loadParams.cpp"
#include "preprocess/grid_pcd.cpp"
#include "methods/linear.cpp"
#include "methods/mrf.cpp"
#include "methods/pwas.cpp"
#include "methods/original.cpp"
#include "postprocess/evaluate.cpp"
#include "postprocess/restore_pcd.cpp"

using namespace std;
using namespace open3d;

void interpolate(int data_no, EnvParams envParams, HyperParams hyperParams,
                 double &time, double &ssim, double &mse, double &mre,
                 bool show_pcd = false, bool show_result = true)
{
    string img_path = envParams.folder_path + to_string(data_no);
    if (envParams.isRGB)
    {
        img_path += "_rgb.png";
    }
    else
    {
        img_path += ".png";
    }
    const string pcd_path = envParams.folder_path + to_string(data_no) + ".pcd";

    auto img = cv::imread(img_path);
    cv::Mat blured;
    cv::GaussianBlur(img, blured, cv::Size(3, 3), 0.5);

    geometry::PointCloud pointcloud;
    auto pcd_ptr = make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(pcd_path, pointcloud))
    {
        cout << "Cannot read" << endl;
    }

    auto start = chrono::system_clock::now();

    vector<vector<double>> original_grid, filtered_grid, original_interpolate_grid, filtered_interpolate_grid;
    vector<vector<int>> target_vs, base_vs;
    *pcd_ptr = pointcloud;
    int layer_cnt = 16;
    calc_grid(pcd_ptr, envParams, original_grid, filtered_grid, original_interpolate_grid, filtered_interpolate_grid, target_vs, base_vs, layer_cnt);

    vector<vector<int>> original_vs;
    if (envParams.isFullHeight)
    {
        original_vs = vector<vector<int>>(envParams.height, vector<int>(envParams.width, 0));
        for (int i = 0; i < envParams.height; i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                original_vs[i][j] = i;
            }
        }
        swap(original_vs, target_vs);
        cout << original_vs.size() << endl;
        cout << target_vs.size() << endl;
    }
    else
    {
        original_vs = target_vs;
    }

    vector<vector<double>> interpolated_z;
    if (envParams.method == "linear")
    {
        linear(interpolated_z, filtered_interpolate_grid, target_vs, base_vs, envParams);
    }
    if (envParams.method == "mrf")
    {
        mrf(interpolated_z, filtered_grid, target_vs, base_vs, envParams, blured,
            hyperParams.mrf_k, hyperParams.mrf_c);
    }
    if (envParams.method == "pwas")
    {
        pwas(interpolated_z, filtered_interpolate_grid, target_vs, base_vs, envParams, blured,
             hyperParams.pwas_sigma_c, hyperParams.pwas_sigma_s,
             hyperParams.pwas_sigma_r, hyperParams.pwas_r);
    }
    if (envParams.method == "original")
    {
        original(interpolated_z, filtered_interpolate_grid, target_vs, base_vs, envParams, blured,
                 hyperParams.original_color_segment_k, hyperParams.original_sigma_s,
                 hyperParams.original_sigma_r, hyperParams.original_r, hyperParams.original_coef_s);
    }

    {
        cv::Mat interpolate_img = cv::Mat::zeros(target_vs.size(), envParams.width, CV_8UC1);
        for (int i = 0; i < target_vs.size(); i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                if (interpolated_z[i][j] > 0)
                {
                    interpolate_img.at<uchar>(i, j) = 255;
                }
            }
        }
        //cv::imshow("hoge", interpolate_img);
        //cv::waitKey();
    }

    { // Evaluate
        time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count();
        evaluate(interpolated_z, original_grid, target_vs, original_vs, envParams, layer_cnt, ssim, mse, mre);
        if (show_result)
        {
            cout << time << "ms" << endl;
            cout << "SSIM = " << fixed << setprecision(5) << ssim << endl;
            cout << "MSE = " << mse << endl;
            cout << "MRE = " << mre << endl;
        }
    }

    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    auto original_ptr = make_shared<geometry::PointCloud>();
    restore_pcd(interpolated_z, original_grid, target_vs, original_vs, envParams, blured, interpolated_ptr, original_ptr);

    if (show_pcd)
    {
        visualization::DrawGeometries({original_ptr, interpolated_ptr}, "Original", 1600, 900);
    }
    if (!io::WritePointCloudToPCD(envParams.folder_path + to_string(data_no) + "_linear.pcd", *interpolated_ptr))
    {
        cout << "Cannot write" << endl;
    }
}