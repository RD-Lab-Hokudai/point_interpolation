#include <vector>
#include <chrono>
#include <experimental/filesystem>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <time.h>

#include "models/envParams.cpp"
#include "models/hyperParams.cpp"
#include "data/loadParams.cpp"
#include "preprocess/grid_pcd.cpp"
#include "preprocess/find_neighbors.cpp"
#include "methods/linear.cpp"
#include "methods/mrf.cpp"
#include "methods/pwas.cpp"
#include "methods/original.cpp"
#include "methods/ip_basic_cv.cpp"
#include "methods/original_cv.cpp"
#include "methods/jbu_cv.cpp"
#include "methods/guided_filter.cpp"
#include "postprocess/evaluate.cpp"
#include "postprocess/generate_depth_image.cpp"
#include "postprocess/restore_pcd.cpp"

using namespace std;
using namespace open3d;
using namespace experimental;

ofstream ofs;

void interpolate_kitti(cv::Mat &interpolated_depth, cv::Mat &img, cv::Mat &depth, EnvParams &envParams, HyperParams &hyperParams)
{
    vector<vector<double>> interpolated(envParams.height, vector<double>(envParams.width, 0));
    vector<vector<double>> filtered(envParams.height, vector<double>(envParams.width, 0));
    vector<vector<int>> vs(envParams.height, vector<int>(envParams.width, 0));
    for (int j = 0; j < envParams.height; j++)
    {
        ushort *p = &depth.at<ushort>(j, 0);
        for (int k = 0; k < envParams.width; k++)
        {
            filtered[j][k] = *p / 256.0;
            vs[j][k] = j;
            p++;
        }
    }

    cv::Mat blured;
    cv::GaussianBlur(img, blured, cv::Size(5, 5), 1.0);

    if (envParams.method == "pwas")
    {
        pwas(interpolated, filtered, vs, vs, envParams, blured,
             hyperParams.pwas_sigma_c, hyperParams.pwas_sigma_s,
             hyperParams.pwas_sigma_r, hyperParams.pwas_r);
    }
    if (envParams.method == "original")
    {
        original(interpolated, filtered, vs, vs, envParams, blured,
                 hyperParams.original_color_segment_k, hyperParams.original_sigma_s,
                 hyperParams.original_sigma_r, hyperParams.original_r, hyperParams.original_coef_s);
    }

    cv::Mat interpolated_inv_depth = cv::Mat::zeros(envParams.height, envParams.width, CV_16UC1);
    for (int j = 0; j < envParams.height; j++)
    {
        ushort *p = &interpolated_inv_depth.at<ushort>(j, 0);
        for (int k = 0; k < envParams.width; k++)
        {
            if (interpolated[j][k] > 0)
            {
                ushort val = 65535 - interpolated[j][k] * 256;
                if (val < 0)
                {
                    val = 0;
                }
                *p = val;
            }
            p++;
        }
    }
    cv::Mat closed;
    cv::Mat full_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::morphologyEx(interpolated_inv_depth, closed, cv::MORPH_CLOSE, full_kernel);
    interpolated_depth = cv::Mat::zeros(envParams.height, envParams.width, CV_16UC1);
    interpolated_depth.forEach<ushort>([&closed](ushort &now, const int position[]) -> void {
        ushort d = closed.at<ushort>(position[0], position[1]);
        if (d > 0)
        {
            now = 65535 - d;
        }
        else
        {
            now = 0;
        }
    });
}

void tune(string img_dir, string depth_dir, string gt_dir, EnvParams &envParams, HyperParams &hyperParams)
{
    vector<string> img_names;
    vector<string> depth_names;
    vector<string> gt_names;
    for (const auto &file : filesystem::directory_iterator(img_dir))
    {
        img_names.emplace_back(file.path().filename());
    }
    for (const auto &file : filesystem::directory_iterator(depth_dir))
    {
        depth_names.emplace_back(file.path().filename());
    }
    for (const auto &file : filesystem::directory_iterator(gt_dir))
    {
        gt_names.emplace_back(file.path().filename());
    }
    sort(img_names.begin(), img_names.end());
    sort(depth_names.begin(), depth_names.end());
    sort(gt_names.begin(), gt_names.end());

    vector<vector<int>> vs(envParams.height, vector<int>(envParams.width, 0));
    for (int i = 0; i < envParams.height; i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            vs[i][j] = i;
        }
    }

    int tuner_cnt = 10;
    vector<cv::Mat> imgs(tuner_cnt);
    vector<vector<vector<double>>> groundtruth(tuner_cnt, vector<vector<double>>(envParams.height, vector<double>(envParams.width, 0)));
    vector<vector<vector<double>>> filtered(tuner_cnt, vector<vector<double>>(envParams.height, vector<double>(envParams.width, 0)));
    for (int i = 0; i < tuner_cnt; i++)
    {
        cv::Mat img = cv::imread(img_dir + img_names[i * 100], IMREAD_ANYCOLOR);
        cv::GaussianBlur(img, imgs[i], cv::Size(5, 5), 1.0);

        cv::Mat gt = cv::imread(gt_dir + gt_names[i * 100], IMREAD_ANYDEPTH);
        cv::Mat depth = cv::imread(depth_dir + depth_names[i * 100], IMREAD_ANYDEPTH);
        for (int j = 0; j < envParams.height; j++)
        {
            ushort *gt_d = &gt.at<ushort>(j, 0);
            ushort *d = &depth.at<ushort>(j, 0);
            for (int k = 0; k < envParams.width; k++)
            {
                groundtruth[i][j][k] = *gt_d / 256.0;
                gt_d++;
                filtered[i][j][k] = *d / 256.0;
                d++;
            }
        }
    }

    {

        ofs = ofstream("original_tuning_kitti.csv", ios::app);
        double best_mse_sum = 1e20;
        double best_color_segment_k = 1;
        double best_sigma_s = 1;
        double best_sigma_r = 1;
        int best_r = 1;
        double best_coef_s = 1;
        // 2020/12/17 MRE : 0.140269	440	1.6	19	7	0.32
        // 2020/12/17 MRE : 0.0597379 820 1.6 inf(ignore) 7 0.03
        //110, 1.6, 19, 7, 0.7
        //270 2 19 7 0.8

        for (double color_segment_k = 100; color_segment_k <= 500; color_segment_k += 10)
        {
            for (double sigma_s = 2; sigma_s <= 2.0; sigma_s += 0.1)
            {
                for (double sigma_r = 19; sigma_r <= 19; sigma_r += 1)
                {
                    for (int r = 7; r <= 7; r += 2)
                    {
                        for (double coef_s = 0.8; coef_s <= 1.0; coef_s += 0.1)
                        {
                            double mse_sum = 0;
                            hyperParams.original_color_segment_k = color_segment_k;
                            hyperParams.original_sigma_s = sigma_s;
                            hyperParams.original_sigma_r = sigma_r;
                            hyperParams.original_r = r;
                            hyperParams.original_coef_s = coef_s;
                            for (int i = 0; i < tuner_cnt; i++)
                            {
                                vector<vector<double>> interpolated;
                                original(interpolated, filtered[i], vs, vs, envParams, imgs[i],
                                         hyperParams.original_color_segment_k, hyperParams.original_sigma_s,
                                         hyperParams.original_sigma_r, hyperParams.original_r, hyperParams.original_coef_s);

                                double mse = 0;
                                for (int j = 0; j < envParams.height; j++)
                                {
                                    for (int k = 0; k < envParams.width; k++)
                                    {
                                        if (groundtruth[i][j][k] <= 0)
                                        {
                                            continue;
                                        }

                                        double delta = (groundtruth[i][j][k] - interpolated[j][k]);
                                        mse += delta * delta;
                                    }
                                }
                                mse_sum += mse;
                            }

                            if (best_mse_sum > mse_sum)
                            {
                                best_mse_sum = mse_sum;
                                best_color_segment_k = color_segment_k;
                                best_sigma_s = sigma_s;
                                best_sigma_r = sigma_r;
                                best_r = r;
                                best_coef_s = coef_s;
                                cout << "Updated : " << mse_sum / tuner_cnt << endl;
                                ofs << mse_sum / tuner_cnt << "," << color_segment_k << "," << sigma_s << "," << sigma_r << "," << r << "," << coef_s << endl;
                            }
                        }
                    }
                }
            }
        }

        cout << "Sigma C = " << best_color_segment_k << endl;
        cout << "Sigma S = " << best_sigma_s << endl;
        cout << "Sigma R = " << best_sigma_r << endl;
        cout << "R = " << best_r << endl;
        cout << "Coef S = " << best_coef_s << endl;
        cout << "Mean error = " << best_mse_sum / tuner_cnt << endl;
    }
}

int main(int argc, char *argv[])
{
    string home = getenv("HOME");
    string img_dir = home + "/Kitti/depth/depth_selection/val_selection_cropped/image/";
    string depth_dir = home + "/Kitti/depth/depth_selection/val_selection_cropped/velodyne_raw/";
    string gt_dir = home + "/Kitti/depth/depth_selection/val_selection_cropped/groundtruth_depth/";
    HyperParams hyperParams = getDefaultHyperParams(true);
    EnvParams envParams = loadParams("");
    envParams.width = 1216;
    envParams.height = 352;
    envParams.method = "original";

    //tune(img_dir, depth_dir, gt_dir, envParams, hyperParams);

    /*
    hyperParams.original_color_segment_k = 270;
    hyperParams.original_sigma_s = 2;
    hyperParams.original_sigma_r = 19;
    hyperParams.original_r = 7;
    hyperParams.original_coef_s = 0.8;
    */

    vector<string> img_names;
    vector<string> depth_names;
    for (const auto &file : filesystem::directory_iterator(img_dir))
    {
        img_names.emplace_back(file.path().filename());
    }
    for (const auto &file : filesystem::directory_iterator(depth_dir))
    {
        depth_names.emplace_back(file.path().filename());
    }
    sort(img_names.begin(), img_names.end());
    sort(depth_names.begin(), depth_names.end());

    double total_time = 0;

    for (int i = 0; i < img_names.size(); i++)
    {
        cv::Mat img = cv::imread(img_dir + img_names[i], IMREAD_ANYCOLOR);
        cv::Mat depth = cv::imread(depth_dir + depth_names[i], IMREAD_ANYDEPTH);

        cv::Mat vs_mat = cv::Mat::zeros(envParams.height, envParams.width, CV_32SC1);
        vs_mat.forEach<int>([](int &now, const int position[]) -> void {
            now = position[0];
        });

        cv::Mat depth_d = cv::Mat::zeros(envParams.height, envParams.width, CV_64FC1);
        depth_d.forEach<double>([&depth](double &now, const int position[]) -> void {
            now = depth.at<ushort>(position[0], position[1]) / 256.0;
        });

        auto start = chrono::system_clock::now();

        cv::Mat blured;
        cv::GaussianBlur(img, blured, cv::Size(5, 5), 1.0);
        cv::Mat target_mat;
        if (envParams.method == "jbu")
        {
            jbu_cv(target_mat, depth_d, vs_mat, vs_mat, envParams, blured,
                   hyperParams.pwas_sigma_s, hyperParams.pwas_sigma_r, hyperParams.pwas_r);
        }
        if (envParams.method == "original")
        {
            original_cv(target_mat, depth_d, vs_mat, vs_mat, envParams, blured,
                        hyperParams.original_color_segment_k, hyperParams.original_sigma_s,
                        hyperParams.original_sigma_r, hyperParams.original_r, hyperParams.original_coef_s);
        }
        if (envParams.method == "guided")
        {
            guided_filter(target_mat, depth_d, vs_mat, vs_mat, envParams, blured);
        }
        if (envParams.method == "ip-basic")
        {
            ip_basic_cv(target_mat, depth_d, vs_mat, vs_mat, envParams);
        }

        double time = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count();

        // 16FC1に変換
        cv::Mat interpolated_depth = cv::Mat::zeros(envParams.height, envParams.width, CV_16UC1);
        interpolated_depth.forEach<ushort>([&target_mat](ushort &now, const int position[]) -> void {
            double d = target_mat.at<double>(position[0], position[1]);
            if (d > 0)
            {
                now = d * 256;
            }
        });

        total_time += time;
        cout << time << "ms" << endl;

        /*
        cv::Mat interpolated_depth;
        interpolate_kitti(interpolated_depth, img, depth, envParams, hyperParams);
*/

        cv::imwrite("../kitti_out_" + envParams.method + "/" + depth_names[i], interpolated_depth);
    }

    cout << "average time : " << total_time / img_names.size() << endl;
}