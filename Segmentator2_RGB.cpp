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

const int width = 640;
const int height = 480;
const double f_x = width;

ofstream ofs;

double PI = acos(-1);
double delta_rad = 0.52698 * PI / 180;
double max_rad = (16.6 + 0.26349) * PI / 180;

struct EnvParams
{
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

class UnionFind
{
    vector<int> par;
    vector<int> elements;

public:
    UnionFind(int length)
    {
        for (int i = 0; i < length; i++)
        {
            par.emplace_back(i);
            elements.emplace_back(1);
        }
    }

    int root(int x)
    {
        int y = x;
        while (par[y] != y)
        {
            y = par[y];
        }
        par[x] = y;
        return y;
    }

    void unite(int x, int y)
    {
        int rx = root(x);
        int ry = root(y);
        if (rx == ry)
        {
            return;
        }

        if (rx > ry)
        {
            swap(rx, ry);
        }
        par[ry] = rx;
        elements[rx] += elements[ry];
    }

    bool same(int x, int y)
    {
        int rx = root(x);
        int ry = root(y);
        return rx == ry;
    }

    int size(int x)
    {
        int rx = root(x);
        return elements[rx];
    }
};

class Graph
{
    vector<tuple<double, int, int>> edges;
    int length;

    double get_diff(cv::Vec3b &a, cv::Vec3b &b)
    {
        double diff = 0;
        for (int i = 0; i < 3; i++)
        {
            diff += (a[i] - b[i]) * (a[i] - b[i]);
        }
        diff = sqrt(diff);
        return diff;
    }

    double get_point_diff(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d a_color, Eigen::Vector3d b_color, double k)
    {
        double diff_normal = 1;
        for (int i = 0; i < 3; i++)
        {
            diff_normal -= abs(a[i] * b[i]);
        }
        double diff_color = (a_color - b_color).norm();
        return diff_normal + k * diff_color;
    }

    double get_threshold(double k, int size)
    {
        return 1.0 * k / size;
    }

public:
    Graph(cv::Mat *img)
    {
        length = img->rows * img->cols;
        int dx[] ={ 1, 0, 0, -1 };
        int dy[] ={ 0, 1, -1, 0 };
        for (int i = 0; i < img->rows; i++)
        {
            cv::Vec3b *row = img->ptr<cv::Vec3b>(i);
            for (int j = 0; j < img->cols; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    int to_x = j + dx[k];
                    int to_y = i + dy[k];
                    if (0 <= to_x && to_x < img->cols && 0 <= to_y && to_y < img->rows)
                    {
                        double diff = get_diff(row[j], img->at<cv::Vec3b>(to_y, to_x));
                        edges.emplace_back(diff, i * img->cols + j, to_y * img->cols + to_x);
                    }
                }
            }
        }
    }

    Graph(shared_ptr<geometry::PointCloud> pcd_ptr, int neighbors, double color_rate)
    {
        length = pcd_ptr->points_.size();
        auto tree = make_shared<geometry::KDTreeFlann>(*pcd_ptr);
        for (int i = 0; i < length; i++)
        {
            vector<int> indexes(neighbors);
            vector<double> dists(neighbors);
            tree->SearchKNN(pcd_ptr->points_[i], neighbors, indexes, dists);
            for (int j = 0; j < indexes.size(); j++)
            {
                int to = indexes[j];
                if (to <= i)
                {
                    continue;
                }

                double diff = get_point_diff(pcd_ptr->normals_[i], pcd_ptr->normals_[to],
                    pcd_ptr->colors_[i], pcd_ptr->colors_[to], color_rate);
                edges.emplace_back(diff, i, to);
            }
        }
    }

    shared_ptr<UnionFind> segmentate(double k, int min_size)
    {
        auto unionFind = make_shared<UnionFind>(length);
        vector<double> thresholds;
        double diff_max = 0;
        double diff_min = 1000000;
        for (int i = 0; i < length; i++)
        {
            thresholds.emplace_back(get_threshold(k, 1));
            double diff = get<0>(edges[i]);
            diff_max = max(diff_max, diff);
            diff_min = min(diff_min, diff);
        }

        /*
        int bucket_len=1000000;
        vector<vector<int>> bucket(bucket_len+1);
        for(int i=0;i<length;i++){
            int diff_level=(int)(bucket_len*(get<0>(edges[i])-diff_min)/(diff_max-diff_min));
            bucket[diff_level].emplace_back(i);
        }

        for (int i = 0; i < bucket.size(); i++)
        {
            for(int j=0;j<bucket[i].size();j++){
            double diff = get<0>(edges[bucket[i][j]]);
            int from = get<1>(edges[bucket[i][j]]);
            int to = get<2>(edges[bucket[i][j]]);

            from = unionFind->root(from);
            to = unionFind->root(to);

            if (from == to)
            {
                continue;
            }

            if (diff <= min(thresholds[from], thresholds[to]))
            {
                unionFind->unite(from, to);
                int root = unionFind->root(from);
                thresholds[root] = diff + get_threshold(k, unionFind->size(root));
            }
            }
        }
        */

        sort(edges.begin(), edges.end());
        for (int i = 0; i < edges.size(); i++)
        {
            double diff = get<0>(edges[i]);
            int from = get<1>(edges[i]);
            int to = get<2>(edges[i]);

            from = unionFind->root(from);
            to = unionFind->root(to);

            if (from == to)
            {
                continue;
            }

            if (diff <= min(thresholds[from], thresholds[to]))
            {
                unionFind->unite(from, to);
                int root = unionFind->root(from);
                thresholds[root] = diff + get_threshold(k, unionFind->size(root));
            }
        }

        for (int i = 0; i < edges.size(); i++)
        {
            int from = get<1>(edges[i]);
            int to = get<2>(edges[i]);
            from = unionFind->root(from);
            to = unionFind->root(to);

            if (unionFind->size(from) <= min_size || unionFind->size(to) <= min_size)
            {
                unionFind->unite(from, to);
            }
        }

        return unionFind;
    }
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
            int u = (int)(width / 2 + f_x * x / z);
            int v = (int)(height / 2 + f_x * y / z);
            if (0 <= u && u < width && 0 <= v && v < height)
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

    original_grid = vector<vector<double>>(64, vector<double>(width, -1));
    filtered_grid = vector<vector<double>>(layer_cnt, vector<double>(width, -1));
    original_interpolate_grid = vector<vector<double>>(64, vector<double>(width, -1));
    filtered_interpolate_grid = vector<vector<double>>(layer_cnt, vector<double>(width, -1));
    vs = vector<vector<int>>(64, vector<int>(width, -1));
    for (int i = 0; i < 64; i++)
    {
        if (all_layers[i].size() == 0)
        {
            continue;
        }

        int now = 0;
        int u0 = (int)(width / 2 + f_x * all_layers[i][0][0] / all_layers[i][0][2]);
        int v0 = (int)(height / 2 + f_x * all_layers[i][0][1] / all_layers[i][0][2]);
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
            int u = (int)(width / 2 + f_x * all_layers[i][j + 1][0] / all_layers[i][j + 1][2]);
            int v = (int)(height / 2 + f_x * all_layers[i][j + 1][1] / all_layers[i][j + 1][2]);
            original_grid[i][u] = all_layers[i][j][2];

            while (now < min(width, u))
            {
                double z = all_layers[i][j][2] + (now - uPrev) * (all_layers[i][j + 1][2] - all_layers[i][j][2]) / (u - uPrev);
                original_interpolate_grid[i][now] = z;
                vs[i][now] = vPrev + (now - uPrev) * (v - vPrev) / (u - uPrev);
                now++;
            }
            uPrev = u;
        }

        int uLast = (int)(width / 2 + f_x * all_layers[i].back()[0] / all_layers[i].back()[2]);
        int vLast = (int)(height / 2 + f_x * all_layers[i].back()[1] / all_layers[i].back()[2]);
        original_grid[i][uLast] = all_layers[i].back()[2];
        while (now < width)
        {
            original_interpolate_grid[i][now] = all_layers[i].back()[2];
            vs[i][now] = vLast;
            now++;
        }
    }
    for (int i = 0; i < layer_cnt; i++)
    {
        for (int j = 0; j < width; j++)
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
            for (int j = 0; j < width; j++)
            {
                double z = original_grid[i][j];
                if (z < 0)
                {
                    continue;
                }
                double x = z * (j - width / 2) / f_x;
                double y = z * (vs[i][j] - height / 2) / f_x;
                original_ptr->points_.emplace_back(x, y, z);
            }

            if (i % (64 / layer_cnt) == 0)
            {
                for (int j = 0; j < width; j++)
                {
                    double z = filtered_interpolate_grid[i / (64 / layer_cnt)][j];
                    if (z < 0)
                    {
                        continue;
                    }
                    double x = z * (j - width / 2) / f_x;
                    double y = z * (vs[i][j] - height / 2) / f_x;
                    filtered_ptr->points_.emplace_back(x, y, z);
                }
            }
        }
        //visualization::DrawGeometries({original_ptr}, "Points", 1200, 720);
    }
}

double segmentate(int data_no, EnvParams envParams, double gaussian_sigma, double color_segment_k, int color_size_min, double sigma_s = 15, double sigma_r = 20, int r = 10, double coef_s = 0.5, bool see_res = false)
{
    const string img_name = envParams.folder_path + to_string(data_no) + "_rgb.png";
    const string file_name = envParams.folder_path + to_string(data_no) + ".pcd";

    auto img = cv::imread(img_name);
    cv::Mat blured;
    cv::GaussianBlur(img, blured, cv::Size(3, 3), gaussian_sigma);

    int length = width * height;
    vector<cv::Vec3b> params_x(length);
    Eigen::VectorXd params_z(length);

    geometry::PointCloud pointcloud;
    auto pcd_ptr = make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(file_name, pointcloud))
    {
        cout << "Cannot read" << endl;
    }

    auto start = chrono::system_clock::now();

    vector<vector<double>> original_grid, filtered_grid, original_interpolate_grid, filtered_interpolate_grid;
    vector<vector<int>> vs;
    *pcd_ptr = pointcloud;
    int layer_cnt = 16;
    calc_grid(pcd_ptr, envParams, original_grid, filtered_grid, original_interpolate_grid, filtered_interpolate_grid, vs, layer_cnt);

    shared_ptr<UnionFind> color_segments;
    {
        Graph graph(&blured);
        color_segments = graph.segmentate(color_segment_k, color_size_min);
        cout<<"Segmentationtime = "<<chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count()<<endl;
        /*
        cv::Mat segment_img=cv::Mat::zeros(height, width, CV_8UC3);
        for (int i=0;i<height;i++) {
            for (int j=0;j<width;j++) {
                int root=color_segments->root(i*width+j);
                segment_img.at<cv::Vec3b>(i, j)=blured.at<cv::Vec3b>(root/width, root%width);
            }
        }
        cv::imshow("original", blured);
        cv::imshow("segments", segment_img);
        cv::waitKey();
        */
    }

    vector<vector<double>> interpolated_z(64, vector<double>(width, 0));
    {
        // Linear interpolation
        for (int i = 0; i + 1 < layer_cnt; i++)
        {
            for (int j = 0; j < width; j++)
            {
                double delta = (filtered_interpolate_grid[i + 1][j] - filtered_interpolate_grid[i][j]) / (64 / layer_cnt);
                double z = filtered_interpolate_grid[i][j];
                for (int k = 0; k < 64 / layer_cnt; k++)
                {
                    interpolated_z[i * (64 / layer_cnt) + k][j] = z;
                    z += delta;
                }
            }
        }
    }

    // Still slow
    {
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < width; j++)
            {
                double coef = 0;
                double val = 0;
                int v = vs[i][j];
                if (v==-1) {
                    continue;
                }
                cv::Vec3b d0 = blured.at<cv::Vec3b>(v, j);
                //cout<<i<<" "<<v*width+j<<endl;
                int r0 = color_segments->root(v * width + j);
                for (int ii = 0; ii < r; ii++)
                {
                    for (int jj = 0; jj < r; jj++)
                    {
                        int dy = ii - r / 2;
                        int dx = jj - r / 2;
                        if (i + dy < 0 || i + dy >= 64 || j + dx < 0 || j + dx >= width)
                        {
                            continue;
                        }

                        int v1 = vs[i + dy][j + dx];
                        if (v1==-1) {
                            continue;
                        }
                        cv::Vec3b d1 = blured.at<cv::Vec3b>(v1, j + dx);
                        int r1 = color_segments->root(v1 * width + j + dx);
                        double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s) * exp(-cv::norm(d0 - d1) /2 / sigma_r / sigma_r);
                        if (r1!=r0) {
                            tmp*=coef_s;
                        }
                        val += tmp * interpolated_z[i + dy][j + dx];
                        coef += tmp;
                    }
                }
                interpolated_z[i][j] = val / coef;
            }
        }
    }

    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    auto filtered_ptr = make_shared<geometry::PointCloud>();
    auto original_ptr = make_shared<geometry::PointCloud>();
    {
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < width; j++)
            {
                double z = interpolated_z[i][j];
                double tanVal = (i - height / 2) / f_x;
                if (original_grid[i][j] <= 0 || z <= 0 /*z < 0 || original_grid[i][j] == 0*/)
                {
                    continue;
                }

                double x = z * (j - width / 2) / f_x;
                double y = z * (vs[i][j] - height / 2) / f_x;

                cv::Vec3b color = blured.at<cv::Vec3b>(vs[i][j], j);
                interpolated_ptr->points_.emplace_back(x, y, z);
                interpolated_ptr->colors_.emplace_back(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0);

                original_ptr->points_.emplace_back(x, y, original_grid[i][j]);
                original_ptr->colors_.emplace_back(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0);

                if (i%(64/layer_cnt)==0) {
                    filtered_ptr->points_.emplace_back(x, y, original_grid[i][j]);
                    filtered_ptr->colors_.emplace_back(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0);
                }
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

            for (int j = 0; j < width; j++)
            {
                if (original_grid[i][j] > 0 && interpolated_z[i][j] > 0)
                {
                    error += abs((original_grid[i][j] - interpolated_z[i][j]) / original_grid[i][j]);
                    cnt++;
                }
            }
        }
        error /= cnt;
        //cout << "cannot cnt = " << (64 - layer_cnt) * width - cnt << endl;
        cout << "Error = " << error << endl;
    }

    { // SSIM evaluation
        double tim = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count();
        cv::Mat original_Mat = cv::Mat::zeros(64 - 64 / layer_cnt + 1, width, CV_64FC1);
        cv::Mat interpolated_Mat = cv::Mat::zeros(64 - 64 / layer_cnt + 1, width, CV_64FC1);
        for (int i = 0; i < 64 - 64 / layer_cnt + 1; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (original_grid[i][j] > 0)
                {
                    original_Mat.at<double>(i, j) = original_grid[i][j];
                    interpolated_Mat.at<double>(i, j) = interpolated_z[i][j];
                }
            }
        }
        double ssim = qm::ssim(original_Mat, interpolated_Mat, 64 / layer_cnt);
        double mse=qm::eqm(original_Mat, interpolated_Mat);
        cout << tim << "ms" << endl;
        cout << "SSIM=" << ssim << endl;
        ofs << data_no << "," << tim << "," << ssim << ","<<mse<<"," << error << "," << endl;
        error = ssim;
    }

    if (see_res)
    {
        Eigen::MatrixXd front(4, 4);
        front << 1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, -1, 0,
            0, 0, 0, 1;
        interpolated_ptr->Transform(front);
        original_ptr->Transform(front);
        filtered_ptr->Transform(front);
        visualization::DrawGeometries({ interpolated_ptr }, "a", 1600, 900);
        //visualization::DrawGeometries({ original_ptr }, "a", 1600, 900);
        //visualization::DrawGeometries({ filtered_ptr }, "a", 1600, 900);
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

    EnvParams params_13jo ={ 498, 485, 509, 481, 517, 500, "../../../data/2020_02_04_13jo/", { 10, 20, 30, 40, 50 }, "res_linear_13jo.csv" };
    EnvParams params_miyanosawa ={ 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", { 700, 1290, 1460, 2350, 3850 }, "res_original_miyanosawa_RGB.csv" };
    EnvParams params_miyanosawa_champ ={ 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", { 1207, 1262, 1264, 1265, 1277 }, "res_original_miyanosawa_RGB.csv" };
    EnvParams params_miyanosawa2 ={ 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_original_miyanosawa_1100-1300_RGB.csv" };

    EnvParams params_miyanosawa_3_3={ 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_original_miyanosawa_0303_1100-1300_RGB.csv" };
    EnvParams params_miyanosawa_3_3_champ ={ 506, 483, 495, 568, 551, 510, "../../../data/2020_03_03_miyanosawa/", { 1207, 1262, 1264, 1265, 1277 }, "res_original_miyanosawa_0303_RGB.csv" };

    EnvParams params_use = params_miyanosawa_3_3;
    ofs = ofstream(params_use.of_name);

    for (int i = 0; i < params_use.data_ids.size(); i++)
    {
        segmentate(params_use.data_ids[i], params_use, 0.5, 110, 1, 1.6, 17, 7, 0.7, false);
    }
    return 0;

    double best_error = 0;
    double best_color_segment_k = 1;
    int best_color_size_min = 1;
    double best_sigma_s = 1;
    double best_sigma_r = 1;
    int best_r = 1;
    double best_coef_s = 0.5;
    // best params 8/10 190 1 90 19 0.9 8 1
    // best params 8/10 0 1 90 19 0.9 3 0.1
    // best params 8/10 110 1 90 1.6 17 7 0.7
    // best params 2020/8/10 110 1 90 1.6 19 7 0.7

    for (double color_segment_k = 100; color_segment_k < 120; color_segment_k += 5)
    {
        for (int color_size_min = 1; color_size_min < 2; color_size_min += 1)
        {
            for (double sigma_s = 1.5; sigma_s < 1.7; sigma_s += 0.01)
            {
                for (double sigma_r = 15; sigma_r < 20; sigma_r += 1)
                {
                    for (int r = 1; r < 9; r+=2)
                    {
                        for (double coef_s = 0; coef_s <= 1; coef_s += 0.1)
                        {
                            cout<<color_segment_k<<" "<<coef_s<<endl;
                            double error = 0;
                            for (int i = 0; i < params_use.data_ids.size(); i++)
                            {
                                error += segmentate(params_use.data_ids[i], params_use, 0.5, color_segment_k, color_size_min, sigma_s, sigma_r, r, coef_s, false);
                            }

                            if (best_error < error)
                            {
                                best_error=error;
                                best_color_segment_k=color_segment_k;
                                best_color_size_min=color_size_min;
                                best_sigma_s = sigma_s;
                                best_sigma_r = sigma_r;
                                best_r = r;
                                best_coef_s=coef_s;
                            }
                        }
                    }
                }
            }
        }
    }

    cout << "Color segment K = " << best_color_segment_k << endl;
    cout << "Color size min = " << best_color_size_min << endl;
    cout << "Sigma S = " << best_sigma_s << endl;
    cout << "Sigma R = " << best_sigma_r << endl;
    cout << "R = " << best_r << endl;
    cout << "Coef S = " << best_coef_s << endl;
    cout << "Mean error = " << best_error / data_nos.size() << endl;
    return 0;
}