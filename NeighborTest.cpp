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

using namespace std;
using namespace open3d;

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
        int dx[] = {1, 0, 0, -1};
        int dy[] = {0, 1, -1, 0};
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

    Graph(shared_ptr<geometry::PointCloud> pcd_ptr, vector<vector<int>> neighbors, double color_rate)
    {
        length = pcd_ptr->points_.size();
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < neighbors[i].size(); j++)
            {
                int to = neighbors[i][j];
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
        for (int i = 0; i < length; i++)
        {
            thresholds.emplace_back(get_threshold(k, 1));
        }

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

// Generic functor
template <typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum
    {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;
};

struct misra1a_functor : Functor<double>
{
    misra1a_functor(int values, double *x, double *y, double *z)
        : inputs_(3), values_(values), x(x), y(y), z(z) {}

    double *x;
    double *y;
    double *z;
    int operator()(const Eigen::VectorXd &b, Eigen::VectorXd &fvec) const
    {
        for (int i = 0; i < values_; ++i)
        {
            fvec[i] = b[0] * x[i] + b[1] * y[i] + z[i] - b[2];
        }
        return 0;
    }
    const int inputs_;
    const int values_;
    int inputs() const { return inputs_; }
    int values() const { return values_; }
};

struct misra1a_functor2 : Functor<double>
{
    misra1a_functor2(int values, double *x, double *y, double *z)
        : inputs_(6), values_(values), x(x), y(y), z(z) {}

    double *x;
    double *y;
    double *z;
    int operator()(const Eigen::VectorXd &b, Eigen::VectorXd &fvec) const
    {
        for (int i = 0; i < values_; ++i)
        {
            fvec[i] = b[0] * x[i] * x[i] + b[1] * y[i] * y[i] + b[2] * x[i] * y[i] + b[3] * x[i] + b[4] * y[i] + z[i] - b[5];
        }
        return 0;
    }
    const int inputs_;
    const int values_;
    int inputs() const { return inputs_; }
    int values() const { return values_; }
};

const int width = 938;
const int height = 606;
//const int width = 882;
//const int height = 560;
const double f_x = width / 2 * 1.01;

// Calibration
double X = 500;
double Y = 474;
double Z = 458;
double theta = 506;
double phi = 527;

shared_ptr<geometry::PointCloud> calc_filtered(shared_ptr<geometry::PointCloud> raw_pcd_ptr,
                                               vector<vector<double>> &base_z, vector<vector<double>> &filtered_z,
                                               vector<vector<int>> &neighbors)
{
    int layer_cnt = 16;

    vector<double> tans;
    double PI = acos(-1);
    double rad = (-16.6 + 0.26349) * PI / 180;
    double delta_rad = 0.52698 * PI / 180;
    double max_rad = (16.6 + 0.26349) * PI / 180;
    while (rad < max_rad + 0.00001)
    {
        tans.emplace_back(tan(rad));
        rad += delta_rad;
    }

    base_z = vector<vector<double>>(height, vector<double>(width));
    filtered_z = vector<vector<double>>(height, vector<double>(width));
    vector<vector<Eigen::Vector3d>> layers(layer_cnt, vector<Eigen::Vector3d>());
    for (int i = 0; i < raw_pcd_ptr->points_.size(); i++)
    {
        double rawX = raw_pcd_ptr->points_[i][1];
        double rawY = -raw_pcd_ptr->points_[i][2];
        double rawZ = -raw_pcd_ptr->points_[i][0];

        double r = sqrt(rawX * rawX + rawZ * rawZ);
        double thetaVal = (theta - 500) / 1000.0;
        double phiVal = (phi - 500) / 1000.0;
        double xp = (rawX * cos(phiVal) - rawY * sin(phiVal)) * cos(thetaVal) - (rawZ * cos(phiVal) - rawY * sin(phiVal)) * sin(thetaVal);
        double yp = rawY * cos(phiVal) + r * sin(phiVal);
        double zp = (rawX * cos(phiVal) - rawY * sin(phiVal)) * sin(thetaVal) + (rawZ * cos(phiVal) - rawY * sin(phiVal)) * cos(thetaVal);
        double x = xp + (X - 500) / 100.0;
        double y = yp + (Y - 500) / 100.0;
        double z = zp + (Z - 500) / 100.0;

        if (z > 0)
        {
            int u = (int)(width / 2 + f_x * x / z);
            int v = (int)(height / 2 + f_x * y / z);
            if (0 <= u && u < width && 0 <= v && v < height)
            {
                auto it = lower_bound(tans.begin(), tans.end(), rawY / r);
                int index = it - tans.begin();
                if (index % (64 / layer_cnt) == 0)
                {
                    layers[index / (64 / layer_cnt)].emplace_back(x, y, z);
                    filtered_z[v][u] = z;
                }
                base_z[v][u] = z;
            }
        }
    }

    int filtered_cnt = 0;
    for (int i = 0; i < layer_cnt; i++)
    {
        sort(begin(layers[i]), end(layers[i]),
             [](Eigen::Vector3d a, Eigen::Vector3d b) { return a[0] / a[2] < b[0] / b[2]; });
        filtered_cnt += layers[i].size();
    }

    neighbors = vector<vector<int>>(filtered_cnt, vector<int>());
    {
        int point_cnt = 0;
        // Find neighbors
        for (int i = 0; i + 1 < layer_cnt; i++)
        {
            for (int j = 0; j < layers[i].size(); j++)
            {
                int u = (int)(width / 2 + f_x * layers[i][j][0] / layers[i][j][2]);
                int v = (int)(height / 2 + f_x * layers[i][j][1] / layers[i][j][2]);
                int u0 = (int)(width / 2 + f_x * layers[i + 1][0][0] / layers[i + 1][0][2]);
                if (u0 > u)
                {
                    int v0 = (int)(height / 2 + f_x * layers[i + 1][0][1] / layers[i + 1][0][2]);
                    int from = point_cnt + j;
                    int to = point_cnt + layers[i].size();

                    neighbors[from].emplace_back(to);
                    neighbors[to].emplace_back(from);
                }
                else
                {
                    int bottom = 0;
                    int top = layers[i + 1].size();
                    while (bottom + 1 < top)
                    {
                        int mid = (bottom + top) / 2;
                        int uTmp = (int)(width / 2 + f_x * layers[i + 1][mid][0] / layers[i + 1][mid][2]);

                        if (uTmp <= u)
                        {
                            bottom = mid;
                        }
                        else
                        {
                            top = mid;
                        }
                    }
                    for (int ii = max(bottom - 1, 0); ii < min(bottom + 2, (int)layers[i + 1].size()); ii++)
                    {
                        int u2 = (int)(width / 2 + f_x * layers[i + 1][ii][0] / layers[i + 1][ii][2]);
                        int v2 = (int)(height / 2 + f_x * layers[i + 1][ii][1] / layers[i + 1][ii][2]);
                        int from = point_cnt + j;
                        int to = point_cnt + layers[i].size() + ii;
                        neighbors[from].emplace_back(to);
                        neighbors[to].emplace_back(from);
                    }
                }
                if (j + 1 < layers[i].size())
                {
                    neighbors[point_cnt + j].emplace_back(point_cnt + j + 1);
                    neighbors[point_cnt + j + 1].emplace_back(point_cnt + j);
                }
                neighbors[point_cnt + j].emplace_back(point_cnt + j); // Contains myself
            }
            point_cnt += layers[i].size();
        }
    }

    auto sorted_ptr = make_shared<geometry::PointCloud>();
    {
        for (int i = 0; i < layer_cnt; i++)
        {
            for (int j = 0; j < layers[i].size(); j++)
            {
                sorted_ptr->points_.emplace_back(layers[i][j]);
            }
        }
    }

    {
        int point_cnt = 0;
        for (int i = 0; i < sorted_ptr->points_.size(); i++)
        {
            Eigen::Vector3d pa = Eigen::Vector3d::Zero();
            for (int j = 0; j < neighbors[i].size(); j++)
            {
                pa += sorted_ptr->points_[neighbors[i][j]];
            }
            pa /= neighbors[i].size();
            Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
            for (int j = 0; j < neighbors[i].size(); j++)
            {
                for (int ii = 0; ii < 3; ii++)
                {
                    for (int jj = 0; jj < 3; jj++)
                    {
                        Q(ii, jj) += (sorted_ptr->points_[neighbors[i][j]][ii] - pa[ii]) * (sorted_ptr->points_[neighbors[i][j]][jj] - pa[jj]);
                    }
                }
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> ES(Q);
            if (ES.info() != Eigen::Success)
            {
                continue;
            }

            sorted_ptr->normals_.emplace_back(ES.eigenvectors().col(0));
        }
    }

    return sorted_ptr;
}

double segmentate(int data_no, double color_segment_k, int color_size_min, double gaussian_sigma, double point_segment_k, int point_size_min, double color_rate, bool see_res = false)
{
    const string img_name = "../../../data/2020_03_03_miyanosawa_img_pcd/" + to_string(data_no) + ".png";
    const string file_name = "../../../data/2020_03_03_miyanosawa_img_pcd/" + to_string(data_no) + ".pcd";
    const bool vertical = true;

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

    vector<vector<double>> base_z, filtered_z;
    vector<vector<int>> neighbors;
    *pcd_ptr = pointcloud;
    shared_ptr<geometry::PointCloud> filtered_ptr = calc_filtered(pcd_ptr, base_z, filtered_z, neighbors);

    { // Coloring
        for (int i = 0; i < filtered_ptr->points_.size(); i++)
        {
            int u = (int)(width / 2 + f_x * filtered_ptr->points_[i][0] / filtered_ptr->points_[i][2]);
            int v = (int)(height / 2 + f_x * filtered_ptr->points_[i][1] / filtered_ptr->points_[i][2]);

            cv::Vec3b color = blured.at<cv::Vec3b>(v, u);
            filtered_ptr->colors_.emplace_back(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0);
        }
    }

    shared_ptr<UnionFind> color_segments;
    {
        auto graph = make_shared<Graph>(&blured);
        color_segments = graph->segmentate(color_segment_k, color_size_min);
    }

    map<int, vector<int>> segments;
    map<int, Eigen::VectorXd> interpolation_params;
    map<int, set<int>> pixel_surface_map;
    { // Point segmentation and interpolation
        Graph graph(filtered_ptr, neighbors, color_rate);
        auto unionFind = graph.segmentate(point_segment_k, point_size_min);

        for (int i = 0; i < filtered_ptr->points_.size(); i++)
        {
            int root = unionFind->root(i);
            segments[root].emplace_back(i);
        }

        cv::Mat bound_img = cv::Mat::zeros(height, width, CV_8UC3);
        for (const auto pair : segments)
        {
            int key = pair.first;
            auto value = pair.second;
            if (value.size() < 3)
            {
                continue;
            }

            for (int i = 0; i < value.size(); i++)
            {
                double x = filtered_ptr->points_[value[i]][0];
                double y = filtered_ptr->points_[value[i]][1];
                double z = filtered_ptr->points_[value[i]][2];
                int u = (int)(width / 2 + f_x * x / z);
                int v = (int)(height / 2 + f_x * y / z);

                int color_root = color_segments->root(v * width + u);
                pixel_surface_map[color_root].insert(key);
            }

            vector<double> xs, ys, zs;
            for (int i = 0; i < value.size(); i++)
            {
                xs.emplace_back(filtered_ptr->points_[value[i]][0]);
                ys.emplace_back(filtered_ptr->points_[value[i]][1]);
                zs.emplace_back(filtered_ptr->points_[value[i]][2]);
            }

            Eigen::VectorXd linear_param(3);
            linear_param << 0, 0, 0;
            misra1a_functor linear_functor(value.size(), &xs[0], &ys[0], &zs[0]);

            Eigen::NumericalDiff<misra1a_functor> linear_numDiff(linear_functor);
            Eigen::LevenbergMarquardt<Eigen::NumericalDiff<misra1a_functor>> linear_lm(linear_numDiff);
            linear_lm.minimize(linear_param);

            if (value.size() >= 6)
            {
                Eigen::VectorXd quadra_param(6);
                quadra_param << 0, 0, 0, 0, 0, 0;
                misra1a_functor2 quadra_functor(value.size(), &xs[0], &ys[0], &zs[0]);
                Eigen::NumericalDiff<misra1a_functor2> quadra_numDiff(quadra_functor);
                Eigen::LevenbergMarquardt<Eigen::NumericalDiff<misra1a_functor2>> quadra_lm(quadra_numDiff);
                quadra_lm.minimize(quadra_param);

                double mre_linear = 0;
                double mre_quadra = 0;
                Eigen::VectorXd linear_fvec(value.size());
                Eigen::VectorXd quadra_fvec(value.size());
                linear_functor(linear_param, linear_fvec);
                quadra_functor(quadra_param, quadra_fvec);
                for (int i = 0; i < value.size(); i++)
                {
                    mre_linear += abs(linear_fvec[i]);
                    mre_quadra += abs(quadra_fvec[i]);
                }

                Eigen::VectorXd res_param(6);
                if (mre_linear < mre_quadra)
                {
                    res_param << 0, 0, 0, linear_param[0], linear_param[1], linear_param[2];
                }
                else
                {
                    res_param = quadra_param;
                }
                interpolation_params.emplace(key, res_param);
            }
            else
            {
                Eigen::VectorXd res_param(6);
                res_param << 0, 0, 0, linear_param[0], linear_param[1], linear_param[2];
                interpolation_params.emplace(key, res_param);
            }
        }
    }

    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    vector<vector<double>> interpolated_z(height, vector<double>(width));
    {
        cv::Mat interpolated_range_img = cv::Mat::zeros(height, width, CV_8UC3);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int root = color_segments->root(i * width + j);

                if (pixel_surface_map.find(root) != pixel_surface_map.end())
                {
                    double best_z = -100;
                    for (auto itr = pixel_surface_map[root].begin(); itr != pixel_surface_map[root].end(); itr++)
                    {
                        auto params = interpolation_params[*itr];
                        double coef_a = (j - width / 2) / f_x;
                        double coef_b = (i - height / 2) / f_x;
                        double a = params[0] * coef_a * coef_a + params[1] * coef_b * coef_b + params[2] * coef_a * coef_b;
                        double b = params[3] * coef_a + params[4] * coef_b + 1;
                        double c = -params[5];

                        double x = 0;
                        double y = 0;
                        double z = 0;

                        if (a == 0)
                        {
                            z = -c / b;
                        }
                        else
                        {
                            // 判別式0未満になりうる これを改善することが鍵か
                            // なぜ0未満になる？
                            // あてはまりそうな面だけ選ぶ？
                            z = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
                        }

                        x = coef_a * z;
                        y = coef_b * z;

                        if (z > 0 && z < 100)
                        {
                            best_z = z;
                            break;
                        }

                        if (z > 0 && z < 100 && abs(interpolated_z[i][j] - z) < abs(interpolated_z[i][j] - best_z))
                        {
                            best_z = z;
                        }
                    }

                    if (best_z != -100)
                    {
                        //cout << interpolated_z[i][j] - best_z << endl;
                        interpolated_z[i][j] = best_z;
                        interpolated_range_img.at<cv::Vec3b>(i, j)[0] = 255;
                    }
                }
            }
        }
        for (int j = 0; j < width; j++)
        {
            vector<int> up(height, -1);
            for (int i = 0; i < height; i++)
            {
                if (interpolated_z[i][j] > 0)
                {
                    up[i] = i;
                }
                else if (i > 0)
                {
                    up[i] = up[i - 1];
                }
            }

            vector<int> down(height, -1);
            for (int i = height - 1; i >= 0; i--)
            {
                if (interpolated_z[i][j] > 0)
                {
                    down[i] = i;
                }
                else if (i + 1 < height)
                {
                    down[i] = down[i + 1];
                }
            }

            for (int i = 0; i < height; i++)
            {
                if (up[i] == -1 && down[i] == -1)
                {
                    interpolated_z[i][j] = -1;
                }
                else if (up[i] == -1 || down[i] == -1 || up[i] == i)
                {
                    interpolated_z[i][j] = interpolated_z[max(up[i], down[i])][j];
                }
                else
                {
                    interpolated_z[i][j] = (interpolated_z[down[i]][j] * (i - up[i]) + interpolated_z[up[i]][j] * (down[i] - i)) / (down[i] - up[i]);
                }
            }
        }
        for (int i = 0; i < height; i++)
        {
            vector<int> left(width, -1);
            for (int j = 0; j < width; j++)
            {
                if (interpolated_z[i][j] > 0)
                {
                    left[j] = j;
                }
                else if (j > 0)
                {
                    left[j] = left[j - 1];
                }
            }

            vector<int> right(width, -1);
            for (int j = width - 1; j >= 0; j--)
            {
                if (interpolated_z[i][j] > 0)
                {
                    right[j] = j;
                }
                else if (j + 1 < width)
                {
                    right[j] = right[j + 1];
                }
            }

            for (int j = 0; j < width; j++)
            {
                if (left[j] == -1 && right[j] == -1)
                {
                    interpolated_z[i][j] = -1;
                }
                else if (left[j] == -1 || right[j] == -1 || left[j] == j)
                {
                    interpolated_z[i][j] = interpolated_z[i][max(left[j], right[j])];
                }
                else
                {
                    interpolated_z[i][j] = (interpolated_z[i][right[j]] * (j - left[j]) + interpolated_z[i][left[j]] * (right[j] - j)) / (right[j] - left[j]);
                }
            }
        }

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                double z = interpolated_z[i][j];
                if (z < 0)
                {
                    continue;
                }

                double x = z * (j - width / 2) / f_x;
                double y = z * (i - height / 2) / f_x;

                double color = blured.at<cv::Vec3b>(i, j)[0] / 255.0;
                interpolated_ptr->points_.emplace_back(x, y, z);
                interpolated_ptr->colors_.emplace_back(color, color, color);
            }
        }
    }

    double error = 0;
    { // Evaluation
        int cnt = 0;
        int cannot_cnt = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (base_z[i][j] > 0 && filtered_z[i][j] == 0 && interpolated_z[i][j] > 0)
                {
                    error += abs((base_z[i][j] - interpolated_z[i][j]) / base_z[i][j]);
                    cnt++;
                }
                if (base_z[i][j] > 0 && filtered_z[i][j] == 0)
                {
                    cannot_cnt++;
                }
            }
        }
        cout << "cannot cnt = " << cannot_cnt - cnt << endl;
        cout << "Error = " << error / cnt << endl;
    }

    if (see_res)
    {
        Eigen::MatrixXd front(4, 4);
        front << 1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, -1, 0,
            0, 0, 0, 1;
        filtered_ptr->Transform(front);
        visualization::DrawGeometries({interpolated_ptr}, "a", 1600, 900);

        cv::imshow("hoge", img);
        cv::waitKey();
    }

    return error;
}

int main(int argc, char *argv[])
{
    vector<int> data_nos = {550, 1000, 1125, 1260, 1550};
    for (int i = 0; i < data_nos.size(); i++)
    {
        segmentate(data_nos[i], 49, 9, 0.5, 19, 3, 1, false);
    }

    double best_error = 100;
    double best_color_segment_k = 1;
    int best_color_size_min = 1;
    double best_point_segment_k = 1;
    double best_color_rate = 0.1;

    for (double color_segment_k = 0; color_segment_k < 10; color_segment_k += 1)
    {
        for (int color_size_min = 0; color_size_min < 10; color_size_min += 1)
        {
            for (double point_segment_k = 0; point_segment_k < 30; point_segment_k += 1)
            {
                for (double color_rate = 0; color_rate < 10; color_rate += 1)
                {
                    double error = 0;
                    for (int i = 0; i < data_nos.size(); i++)
                    {
                        error += segmentate(data_nos[i], color_segment_k, color_size_min, 0.5, point_segment_k, 3, color_rate);
                    }
                    error /= data_nos.size();

                    if (best_error > error)
                    {
                        best_error = error;
                        best_color_segment_k = color_segment_k;
                        best_color_size_min = color_size_min;
                        best_point_segment_k = point_segment_k;
                        best_color_rate = color_rate;
                        cout << color_segment_k << color_size_min << point_segment_k << color_rate << endl;
                        cout << "Error = " << error << endl;
                    }
                }
            }
        }
    }

    cout << "Color segment k = " << best_color_segment_k << endl;
    cout << "Color size min = " << best_color_size_min << endl;
    cout << "Point segment k = " << best_point_segment_k << endl;
    cout << "Color rate = " << best_color_rate << endl;
    return 0;
}