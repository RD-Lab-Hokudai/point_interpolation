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

#include "quality_metrics_OpenCV.cpp"

using namespace std;
using namespace open3d;

ofstream ofs;

struct EnvParams
{
    int width;
    int height;
    double f_xy;
    int X;
    int Y;
    int Z;
    int roll;
    int pitch;
    int yaw;

    string folder_path;
    vector<int> data_ids;

    string of_name;
};

void calc_grid(shared_ptr<geometry::PointCloud> raw_pcd_ptr, EnvParams envParams,
               vector<vector<double>> &original_grid, vector<vector<double>> &filtered_grid,
               vector<vector<double>> &original_interpolate_grid, vector<vector<double>> &filtered_interpolate_grid,
               vector<vector<int>> &vs, int layer_cnt = 16)
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
    vs = vector<vector<int>>(64, vector<int>(envParams.width, -1));
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
            vs[i][now] = v0;
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
                double z = all_layers[i][j][2] + (now - uPrev) * (all_layers[i][j + 1][2] - all_layers[i][j][2]) / (u - uPrev);
                original_interpolate_grid[i][now] = z;
                vs[i][now] = vPrev + (now - uPrev) * (v - vPrev) / (u - uPrev);
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
            vs[i][now] = vLast;
            now++;
        }
    }
    for (int i = 0; i < layer_cnt; i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            filtered_grid[i][j] = original_grid[i * (64 / layer_cnt)][j];
            filtered_interpolate_grid[i][j] = original_interpolate_grid[i * (64 / layer_cnt)][j];
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
                double y = z * (vs[i][j] - envParams.height / 2) / envParams.f_xy;
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
                    double y = z * (vs[i][j] - envParams.height / 2) / envParams.f_xy;
                    filtered_ptr->points_.emplace_back(x, y, z);
                }
            }
        }
        //visualization::DrawGeometries({original_ptr}, "Points", 1200, 720);
    }
}

double segmentate(int data_no, EnvParams envParams, bool see_res = false)
{
    const string img_path = envParams.folder_path + to_string(data_no) + "_rgb.png";
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
    vector<vector<int>> vs;
    *pcd_ptr = pointcloud;
    int layer_cnt = 16;
    calc_grid(pcd_ptr, envParams, original_grid, filtered_grid, original_interpolate_grid, filtered_interpolate_grid, vs, layer_cnt);

    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    vector<vector<double>> interpolated_z(64, vector<double>(envParams.width, 0));
    {
        // Linear interpolation
        for (int i = 0; i + 1 < layer_cnt; i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                double delta = (filtered_interpolate_grid[i + 1][j] - filtered_interpolate_grid[i][j]) / (64 / layer_cnt);
                double z = filtered_interpolate_grid[i][j];
                for (int k = 0; k <= 64 / layer_cnt; k++)
                {
                    interpolated_z[i * (64 / layer_cnt) + k][j] = z;
                    z += delta;
                }
            }
        }
    }

    {
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                double z = interpolated_z[i][j];
                if (original_grid[i][j] <= 0 || z <= 0 /*z < 0 || original_grid[i][j] == 0*/)
                {
                    continue;
                }

                double x = z * (j - envParams.width / 2) / envParams.f_xy;
                double y = z * (vs[i][j] - envParams.height / 2) / envParams.f_xy;
                interpolated_ptr->points_.emplace_back(x, y, z);
                cv::Vec3b color = img.at<cv::Vec3b>(vs[i][j], j);
                interpolated_ptr->colors_.emplace_back(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0);
            }
        }
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
        cv::Mat original_Mat = cv::Mat::zeros(64 - 64 / layer_cnt + 1, envParams.width, CV_64FC1);
        cv::Mat interpolated_Mat = cv::Mat::zeros(64 - 64 / layer_cnt + 1, envParams.width, CV_64FC1);
        cv::Mat original_interpolated_Mat = cv::Mat::zeros(64 - 64 / layer_cnt + 1, envParams.width, CV_64FC1);
        for (int i = 0; i < 64 - 64 / layer_cnt + 1; i++)
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
        double ssim = qm::ssim(original_interpolated_Mat, interpolated_Mat, 64 / layer_cnt);
        double mse = qm::eqm(original_interpolated_Mat, interpolated_Mat);
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
    for (int i = 1100; i <= 1300; i++)
    {
        data_nos.emplace_back(i);
    }

    // Calibration
    // 03_03_miyanosawa
    /*
int X = 500;
int Y = 474;
int Z = 458;
int theta = 506;
int phi = 527;
*/

    EnvParams params_13jo = {938, 606, 938 / 2 * 1.01, 498, 485, 509, 481, 517, 500, "../../../data/2020_02_04_13jo/", {10, 20, 30, 40, 50}, "res_linear_13jo.csv"};
    EnvParams params_miyanosawa = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", {700, 1290, 1460, 2350, 3850}, "res_linear_miyanosawa.csv"};
    EnvParams params_miyanosawa_champ = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", {1207, 1262, 1264, 1265, 1277}, "res_linear_miyanosawa_RGB.csv"};
    EnvParams params_miyanosawa2 = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_linear_miyanosawa_1100-1300_RGB.csv"};

    EnvParams params_miyanosawa_3_3 = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_linear_miyanosawa_0303_1100-1300_RGB.csv"};

    EnvParams params_use = params_miyanosawa_3_3;
    ofs = ofstream(params_use.of_name);

    for (int i = 0; i < params_use.data_ids.size(); i++)
    {
        segmentate(params_use.data_ids[i], params_use, false);
    }
}