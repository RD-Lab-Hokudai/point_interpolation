#include <iostream>
#include <vector>
#include <stack>
#include <map>
#include <set>
#include <chrono>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <time.h>

#include "quality_metrics_OpenCV_2.cpp"
#include "models/envParams.cpp"
#include "methods/linear.cpp"

using namespace std;
using namespace open3d;

ofstream ofs;

void calc_grid(shared_ptr<geometry::PointCloud> raw_pcd_ptr, EnvParams envParams,
               vector<vector<double>> &original_grid, vector<vector<double>> &filtered_grid,
               vector<vector<double>> &original_interpolate_grid, vector<vector<double>> &filtered_interpolate_grid,
               vector<vector<int>> &target_vs, vector<vector<int>> &base_vs, int layer_cnt = 16)
{
    vector<double> tans;
    double PI = acos(-1);
    double delta_rad = 0.52698 * PI / 180;
    double max_rad = (16.6 + 0.26349) * PI / 180;
    double rad = (-16.6 + 0.26349) * PI / 180;
    while (rad < max_rad + 0.00001)
    {
        tans.emplace_back(tan(rad));
        rad += delta_rad;
    }

    vector<vector<Eigen::Vector3d>> all_layers(64, vector<Eigen::Vector3d>());
    for (int i = 0; i < raw_pcd_ptr->points_.size(); i++)
    {
        double rawX = raw_pcd_ptr->points_[i][1];
        double rawY = -raw_pcd_ptr->points_[i][2];
        double rawZ = -raw_pcd_ptr->points_[i][0];

        double r = sqrt(rawX * rawX + rawZ * rawZ);
        double rollVal = (envParams.roll - 500) / 1000.0;
        double pitchVal = (envParams.pitch - 500) / 1000.0;
        double yawVal = (envParams.yaw - 500) / 1000.0;
        double xp = cos(yawVal) * cos(pitchVal) * rawX + (cos(yawVal) * sin(pitchVal) * sin(rollVal) - sin(yawVal) * cos(rollVal)) * rawY + (cos(yawVal) * sin(pitchVal) * cos(rollVal) + sin(yawVal) * sin(rollVal)) * rawZ;
        double yp = sin(yawVal) * cos(pitchVal) * rawX + (sin(yawVal) * sin(pitchVal) * sin(rollVal) + cos(yawVal) * cos(rollVal)) * rawY + (sin(yawVal) * sin(pitchVal) * cos(rollVal) - cos(yawVal) * sin(rollVal)) * rawZ;
        double zp = -sin(pitchVal) * rawX + cos(pitchVal) * sin(rollVal) * rawY + cos(pitchVal) * cos(rollVal) * rawZ;
        double x = xp + (envParams.X - 500) / 100.0;
        double y = yp + (envParams.Y - 500) / 100.0;
        double z = zp + (envParams.Z - 500) / 100.0;

        if (z > 0)
        {
            int u = (int)(envParams.width / 2 + envParams.f_xy * x / z);
            int v = (int)(envParams.height / 2 + envParams.f_xy * y / z);
            if (0 <= u && u < envParams.width && 0 <= v && v < envParams.height)
            {
                auto it = lower_bound(tans.begin(), tans.end(), rawY / r);
                int index = it - tans.begin();
                all_layers[index].emplace_back(x, y, z);
            }
        }
    }

    for (int i = 0; i < 64; i++)
    {
        // no sort
        vector<Eigen::Vector3d> removed;
        for (size_t j = 0; j < all_layers[i].size(); j++)
        {
            while (removed.size() > 0 && removed.back()[0] * all_layers[i][j][2] >= all_layers[i][j][0] * removed.back()[2])
            {
                removed.pop_back();
            }
            removed.emplace_back(all_layers[i][j]);
        }
    }

    original_grid = vector<vector<double>>(64, vector<double>(envParams.width, -1));
    filtered_grid = vector<vector<double>>(layer_cnt, vector<double>(envParams.width, -1));
    original_interpolate_grid = vector<vector<double>>(64, vector<double>(envParams.width, -1));
    filtered_interpolate_grid = vector<vector<double>>(layer_cnt, vector<double>(envParams.width, -1));
    target_vs = vector<vector<int>>(64, vector<int>(envParams.width, -1));
    base_vs = vector<vector<int>>(layer_cnt, vector<int>(envParams.width, -1));

    for (int i = 0; i < 64; i++)
    {
        if (all_layers[i].size() == 0)
        {
            continue;
        }

        int now = 0;
        int u0 = (int)(envParams.width / 2 + envParams.f_xy * all_layers[i][0][0] / all_layers[i][0][2]);
        int v0 = (int)(envParams.height / 2 + envParams.f_xy * all_layers[i][0][1] / all_layers[i][0][2]);
        while (now < u0)
        {
            original_interpolate_grid[i][now] = all_layers[i][0][2];
            target_vs[i][now] = v0;
            now++;
        }
        int uPrev = u0;
        int vPrev = v0;
        for (int j = 0; j + 1 < all_layers[i].size(); j++)
        {
            int u = (int)(envParams.width / 2 + envParams.f_xy * all_layers[i][j + 1][0] / all_layers[i][j + 1][2]);
            int v = (int)(envParams.height / 2 + envParams.f_xy * all_layers[i][j + 1][1] / all_layers[i][j + 1][2]);
            original_grid[i][u] = all_layers[i][j][2];

            while (now < min(envParams.width, u))
            {
                double angle = (all_layers[i][j + 1][2] - all_layers[i][j][2]) / (all_layers[i][j + 1][0] - all_layers[i][j][0]);
                double tan = (now - envParams.width / 2) / envParams.f_xy;
                double z = (all_layers[i][j][2] - angle * all_layers[i][j][0]) / (1 - tan * angle);
                original_interpolate_grid[i][now] = z;
                target_vs[i][now] = vPrev + (now - uPrev) * (v - vPrev) / (u - uPrev);
                now++;
            }
            uPrev = u;
        }

        int uLast = (int)(envParams.width / 2 + envParams.f_xy * all_layers[i].back()[0] / all_layers[i].back()[2]);
        int vLast = (int)(envParams.height / 2 + envParams.f_xy * all_layers[i].back()[1] / all_layers[i].back()[2]);
        original_grid[i][uLast] = all_layers[i].back()[2];
        while (now < envParams.width)
        {
            original_interpolate_grid[i][now] = all_layers[i].back()[2];
            target_vs[i][now] = vLast;
            now++;
        }
    }
    for (int i = 0; i < layer_cnt; i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            filtered_grid[i][j] = original_grid[i * (64 / layer_cnt)][j];
            filtered_interpolate_grid[i][j] = original_interpolate_grid[i * (64 / layer_cnt)][j];
            base_vs[i][j] = target_vs[i * (64 / layer_cnt)][j];
        }
    }

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
                double y = z * (target_vs[i][j] - envParams.height / 2) / envParams.f_xy;
                original_ptr->points_.emplace_back(x, y, z);
            }

            if (i % (64 / layer_cnt) == 0)
            {
                for (int j = 0; j < envParams.width; j++)
                {
                    double z = filtered_interpolate_grid[i / (64 / layer_cnt)][j];
                    if (z < 0)
                    {
                        continue;
                    }
                    double x = z * (j - envParams.width / 2) / envParams.f_xy;
                    double y = z * (target_vs[i][j] - envParams.height / 2) / envParams.f_xy;
                    filtered_ptr->points_.emplace_back(x, y, z);
                }
            }
        }
        //visualization::DrawGeometries({original_ptr}, "Points", 1200, 720);
    }
}

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
    cout << "before gridding" << endl;
    calc_grid(pcd_ptr, envParams, original_grid, filtered_grid, original_interpolate_grid, filtered_interpolate_grid, target_vs, base_vs, layer_cnt);

    cout << "after gridding" << endl;
    vector<vector<double>> interpolated_z(64, vector<double>(envParams.width, 0));
    linear(interpolated_z, filtered_grid, target_vs, base_vs, envParams);
    cout << "after interpolation" << endl;

    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    {
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                double z = interpolated_z[i][j];
                if (/*original_grid[i][j] <= 0 ||*/ z <= 0 || target_vs[i][j] < 0)
                {
                    continue;
                }

                double x = z * (j - envParams.width / 2) / envParams.f_xy;
                double y = z * (target_vs[i][j] - envParams.height / 2) / envParams.f_xy;
                interpolated_ptr->points_.emplace_back(x, y, z);
                cv::Vec3b color = img.at<cv::Vec3b>(target_vs[i][j], j);
                interpolated_ptr->colors_.emplace_back(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0);
            }
        }
    }

    {
        auto original_colored_ptr = make_shared<geometry::PointCloud>();
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                double z = original_interpolate_grid[i][j];
                if (z <= 0 || target_vs[i][j] < 0)
                {
                    continue;
                }

                double x = z * (j - envParams.width / 2) / envParams.f_xy;
                double y = z * (target_vs[i][j] - envParams.height / 2) / envParams.f_xy;
                original_colored_ptr->points_.emplace_back(x, z, -y);
                cv::Vec3b color = img.at<cv::Vec3b>(target_vs[i][j], j);
                original_colored_ptr->colors_.emplace_back(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0);
            }
        }
        visualization::DrawGeometries({original_colored_ptr}, "Original", 1600, 900);
        visualization::DrawGeometries({interpolated_ptr}, "Original", 1600, 900);
        if (!io::WritePointCloudToPCD(envParams.folder_path + to_string(data_no) + "_linear.pcd", *original_colored_ptr))
        {
            cout << "Cannot write" << endl;
        }
        /*
        geometry::PointCloud hoge;
        auto hoge_ptr = make_shared<geometry::PointCloud>();
        if (!io::ReadPointCloud(envParams.folder_path + to_string(data_no) + "_color.pcd", hoge))
        {
            cout << "Cannot read" << endl;
        }
        *hoge_ptr = hoge;
        visualization::DrawGeometries({hoge_ptr}, "Read", 1600, 900);
        */
    }

    double error = 0;
    { // Evaluation
        int cnt = 0;
        int cannot_cnt = 0;
        for (int i = 0; i < 64; i++)
        {
            if (i % (64 / layer_cnt) == 0)
            {
                continue;
            }

            for (int j = 0; j < envParams.width; j++)
            {
                if (original_grid[i][j] > 0 && interpolated_z[i][j] > 0)
                {
                    error += abs((original_grid[i][j] - interpolated_z[i][j]) / original_grid[i][j]);
                    cnt++;
                }
            }
        }
        error /= cnt;
        cout << "Error = " << error << endl;
    }

    { // SSIM evaluation
        double tim = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count();
        cv::Mat original_Mat = cv::Mat::zeros(64, envParams.width, CV_64FC1);
        cv::Mat interpolated_Mat = cv::Mat::zeros(64, envParams.width, CV_64FC1);
        cv::Mat original_interpolated_Mat = cv::Mat::zeros(64, envParams.width, CV_64FC1);
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                if (original_grid[i][j] > 0)
                {
                    original_Mat.at<double>(i, j) = original_grid[i][j];
                    interpolated_Mat.at<double>(i, j) = interpolated_z[i][j];
                }

                if (original_interpolate_grid[i][j] > 0)
                {
                    original_interpolated_Mat.at<double>(i, j) = original_interpolate_grid[i][j];
                    interpolated_Mat.at<double>(i, j) = interpolated_z[i][j];
                }
            }
        }
        double ssim = qm::ssim(original_Mat, interpolated_Mat, 64 / layer_cnt);
        double mse = qm::eqm(original_Mat, interpolated_Mat);
        cout << tim << "ms" << endl;
        cout << "SSIM=" << ssim << endl;
        ofs << data_no << "," << tim << "," << ssim << "," << mse << "," << error << "," << endl;
    }

    if (see_res)
    {
        Eigen::MatrixXd front(4, 4);
        front << 1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, -1, 0,
            0, 0, 0, 1;
        pcd_ptr->Transform(front);
        interpolated_ptr->Transform(front);
        visualization::DrawGeometries({pcd_ptr}, "b", 1600, 900);
        visualization::DrawGeometries({interpolated_ptr}, "a", 1600, 900);
    }

    return error;
}

int main(int argc, char *argv[])
{
    //vector<int> data_nos = {550, 1000, 1125, 1260, 1550}; // 03_03_miyanosawa
    //vector<int> data_nos = {10, 20, 30, 40, 50}; // 02_04_13jo
    //vector<int> data_nos = {700, 1290, 1460, 2350, 3850}; // 02_04_miyanosawa

    vector<int> data_nos;
    for (int i = 100; i <= 300; i++)
    {
        data_nos.emplace_back(i);
    }

    EnvParams params_13jo = {938, 606, 938 / 2 * 1.01, 498, 485, 509, 481, 517, 500, "../../../data/2020_02_04_13jo/", {10, 20, 30, 40, 50}, "res_linear_13jo.csv", "linear", false, true};
    EnvParams params_miyanosawa = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", {700, 1290, 1460, 2350, 3850}, "res_linear_miyanosawa.csv", "linear", false, true};
    EnvParams params_miyanosawa_champ = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", {1207, 1262, 1264, 1265, 1277}, "res_linear_miyanosawa_RGB.csv", "linear", false, true};
    EnvParams params_miyanosawa2 = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_linear_miyanosawa_1100-1300_RGB.csv", "linear", false, true};

    EnvParams params_miyanosawa_3_3 = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_linear_miyanosawa_0303_1100-1300_RGB.csv", "linear", false, true};

    EnvParams params_use = params_miyanosawa_3_3;
    ofs = ofstream(params_use.of_name);

    for (int i = 0; i < params_use.data_ids.size(); i++)
    {
        segmentate(params_use.data_ids[i], params_use, false);
    }
}