#include <iostream>
#include <vector>
#include <chrono>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <time.h>

#include "models/envParams.cpp"
#include "preprocess/grid_pcd.cpp"
#include "methods/linear.cpp"
#include "methods/mrf.cpp"
#include "methods/pwas.cpp"
#include "methods/original.cpp"
#include "quality_metrics_OpenCV_2.cpp"
#include "postprocess/evaluate.cpp"
#include "postprocess/restore_pcd.cpp"

using namespace std;
using namespace open3d;

ofstream ofs;

double segmentate(int data_no, EnvParams envParams, bool see_res = false)
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
        mrf(interpolated_z, filtered_interpolate_grid, target_vs, base_vs, envParams, blured);
    }
    if (envParams.method == "pwas")
    {
        pwas(interpolated_z, filtered_interpolate_grid, target_vs, base_vs, envParams, blured);
        cout << "aaa" << endl;
    }
    if (envParams.method == "original")
    {
        original(interpolated_z, filtered_interpolate_grid, target_vs, base_vs, envParams, blured);
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

    double error = 0;
    { // Evaluate
        double tim = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count();
        double ssim, mse, mre;
        evaluate(interpolated_z, original_grid, target_vs, original_vs, envParams, layer_cnt, ssim, mse, mre);

        cout << tim << "ms" << endl;
        cout << "SSIM = " << fixed << setprecision(5) << ssim << endl;
        cout << "MSE = " << mse << endl;
        cout << "MRE = " << mre << endl;
        ofs << data_no << "," << tim << "," << ssim << "," << mse << "," << mre << "," << endl;

        error = mre;
    }

    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    auto original_ptr = make_shared<geometry::PointCloud>();
    restore_pcd(interpolated_z, original_grid, target_vs, original_vs, envParams, blured, interpolated_ptr, original_ptr);

    if (see_res)
    {
        visualization::DrawGeometries({original_ptr, interpolated_ptr}, "Original", 1600, 900);
    }
    if (!io::WritePointCloudToPCD(envParams.folder_path + to_string(data_no) + "_linear.pcd", *interpolated_ptr))
    {
        cout << "Cannot write" << endl;
    }

    return error;
}

int main(int argc, char *argv[])
{
    vector<int> data_nos;
    for (int i = 1100; i <= 1300; i++)
    {
        data_nos.emplace_back(i);
    }

    EnvParams params_13jo = {938, 606, 938 / 2 * 1.01, 498, 485, 509, 481, 517, 500, "../../../data/2020_02_04_13jo/", {10, 20, 30, 40, 50}, "res_linear_13jo.csv", "linear", true, false};
    EnvParams params_miyanosawa = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", {700, 1290, 1460, 2350, 3850}, "res_linear_miyanosawa.csv", "linear", false, true};
    EnvParams params_miyanosawa_champ = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", {1207, 1262, 1264, 1265, 1277}, "res_linear_miyanosawa_RGB.csv", "linear", false, true};

    EnvParams params_miyanosawa_3_3 = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_linear_miyanosawa_0303_1100-1300_RGB.csv", "linear", false, true};
    EnvParams params_miyanosawa_3_3_pwas = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_pwas_miyanosawa_0303_1100-1300_RGB.csv", "pwas", false, true};
    EnvParams params_miyanosawa_3_3_pwas_champ = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", {1207, 1262, 1264, 1265, 1277}, "res_pwas_miyanosawa_0303_RGB.csv", "pwas", false, true};
    EnvParams params_miyanosawa_3_3_original = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_original_miyanosawa_0303_1100-1300_RGB.csv", "original", false, true};

    EnvParams params_miyanosawa_3_3_thermal = {938, 606, 938 / 2 * 1.01, 495, 466, 450, 469, 503, 487, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_linear_miyanosawa_0303_1100-1300_Thermal.csv", "linear", false, false};
    EnvParams params_miyanosawa_3_3_thermal_pwas = {938, 606, 938 / 2 * 1.01, 495, 466, 450, 469, 503, 487, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_pwas_miyanosawa_0303_1100-1300_Thermal.csv", "pwas", false, false};
    EnvParams params_miyanosawa_3_3_thermal_original = {938, 606, 938 / 2 * 1.01, 495, 466, 450, 469, 503, 487, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_original_miyanosawa_0303_1100-1300_Thermal.csv", "original", false, false};

    EnvParams params_miyanosawa_0204_rgb_linear = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_linear_miyanosawa_0204_1100-1300_RGB.csv", "linear", false, true};
    EnvParams params_miyanosawa_0204_rgb_mrf = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_mrf_miyanosawa_0204_1100-1300_RGB.csv", "mrf", false, true};
    EnvParams params_miyanosawa_0204_rgb_pwas = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_pwas_miyanosawa_0204_1100-1300_RGB.csv", "pwas", false, true};
    EnvParams params_miyanosawa_0204_rgb_original = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_original_miyanosawa_0204_1100-1300_RGB.csv", "original", false, true};

    EnvParams params_miyanosawa_0204_thermal_linear = {938, 606, 938 / 2 * 1.01, 495, 475, 458, 488, 568, 500, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_linear_miyanosawa_0204_1100-1300_Thermal.csv", "linear", false, false};
    EnvParams params_miyanosawa_0204_thermal_mrf = {938, 606, 938 / 2 * 1.01, 495, 475, 458, 488, 568, 500, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_mrf_miyanosawa_0204_1100-1300_Thermal.csv", "mrf", false, false};
    EnvParams params_miyanosawa_0204_thermal_pwas = {938, 606, 938 / 2 * 1.01, 495, 475, 458, 488, 568, 500, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_pwas_miyanosawa_0204_1100-1300_Thermal.csv", "pwas", false, false};
    EnvParams params_miyanosawa_0204_thermal_original = {938, 606, 938 / 2 * 1.01, 495, 475, 458, 488, 568, 500, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_original_miyanosawa_0204_1100-1300_Thermal.csv", "original", false, false};

    EnvParams params_use = params_miyanosawa_0204_rgb_original;
    ofs = ofstream(params_use.of_name);

    for (int i = 0; i < params_use.data_ids.size(); i++)
    {
        segmentate(params_use.data_ids[i], params_use, true);
    }
    return 0;

    params_use = params_miyanosawa_3_3_pwas_champ;
    double best_error = 1000000;
    double best_sigma_c = 1;
    double best_sigma_s = 1;
    double best_sigma_r = 1;
    int best_r = 1;
    // best params 2020/08/03 sigma_c:1000 sigma_s:1.99 sigma_r:19 r:7
    // best params 2020/08/10 sigma_c:12000 sigma_s:1.6 sigma_r:19 r:7
    // best params 2020/08/10 sigma_c:8000 sigma_s:1.6 sigma_r:19 r:7

    for (double sigma_c = 10; sigma_c <= 1000; sigma_c += 10)
    {
        for (double sigma_s = 0.1; sigma_s < 1.7; sigma_s += 0.1)
        {
            for (double sigma_r = 1; sigma_r < 100; sigma_r += 10)
            {
                for (int r = 1; r < 9; r += 2)
                {
                    double error = 0;
                    for (int i = 0; i < params_use.data_ids.size(); i++)
                    {
                        error += segmentate(params_use.data_ids[i], params_use);
                    }

                    if (best_error > error)
                    {
                        best_error = error;
                        best_sigma_c = sigma_c;
                        best_sigma_s = sigma_s;
                        best_sigma_r = sigma_r;
                        best_r = r;
                    }
                }
            }
        }
    }

    cout << "Sigma C = " << best_sigma_c << endl;
    cout << "Sigma S = " << best_sigma_s << endl;
    cout << "Sigma R = " << best_sigma_r << endl;
    cout << "R = " << best_r << endl;
    cout << "Mean error = " << best_error / params_use.data_ids.size() << endl;
}