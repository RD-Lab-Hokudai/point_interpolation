#include <iostream>
#include <vector>
#include <stack>
#include <map>
#include <set>
#include <chrono>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
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
                        double diff = get_diff(img->at<cv::Vec3b>(i, j), img->at<cv::Vec3b>(to_y, to_x));
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

double det(Eigen::Vector2i a, Eigen::Vector2i b)
{
    return a[0] * b[1] - a[1] * b[0];
}

Eigen::Vector2i sub(Eigen::Vector2i a, Eigen::Vector2i b)
{
    Eigen::Vector2i res(a - b);
    return res;
}

vector<Eigen::Vector2i> get_convex_hull(vector<Eigen::Vector2i> uvs)
{
    sort(uvs.begin(), uvs.end(), [](Eigen::Vector2i a, Eigen::Vector2i b) {
        if (a[0] == b[0])
        {
            return a[1] < b[1];
        }
        return a[0] < b[0];
    });

    int size_convex_hull = 0;
    vector<Eigen::Vector2i> ch;
    for (int i = 0; i < uvs.size(); i++)
    {
        while (size_convex_hull > 1)
        {
            Eigen::Vector2i current = sub(ch[size_convex_hull - 1], ch[size_convex_hull - 2]);
            Eigen::Vector2i target = sub(uvs[i], ch[size_convex_hull - 2]);
            if (det(current, target) > 0)
            {
                break;
            }
            size_convex_hull--;
            ch.pop_back();
        }
        ch.emplace_back(uvs[i]);
        size_convex_hull++;
    }

    int t = size_convex_hull;
    for (int i = uvs.size() - 2; i > -1; i--)
    {
        while (size_convex_hull > t)
        {
            Eigen::Vector2i current = sub(ch[size_convex_hull - 1], ch[size_convex_hull - 2]);
            Eigen::Vector2i target = sub(uvs[i], ch[size_convex_hull - 2]);
            if (det(current, target) > 0)
            {
                break;
            }
            size_convex_hull--;
            ch.pop_back();
        }
        ch.emplace_back(uvs[i]);
        size_convex_hull++;
    }

    ch.pop_back();
    return ch;
}

double segmentation(double color_segment_k, int color_size_min, double gaussian_sigma, int point_neighbors, double point_segment_k, int point_size_min, double color_rate)
{
    auto start = chrono::system_clock::now();

    const string file_name = "../2271.pcd";
    const string img_path = "../2271.png";

    auto img = cv::imread(img_path);
    const int width = img.cols;
    const int height = img.rows;
    const double f_x = width / 2 * 1.01;

    shared_ptr<UnionFind> color_segments;
    { // Image segmentation
        auto blured = cv::Mat();
        cv::GaussianBlur(img, blured, cv::Size(3, 3), gaussian_sigma);
        img = blured;
        auto graph = make_shared<Graph>(&img);
        color_segments = graph->segmentate(color_segment_k, color_size_min);
        int segment_cnt = 0;
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                int parent = color_segments->root(i * width + j);
                if (parent == i * img.cols + j)
                {
                    segment_cnt++;
                }
                auto color = img.at<cv::Vec3b>(parent / width, parent % width);
                img.at<cv::Vec3b>(i, j)[0] = color[0];
                img.at<cv::Vec3b>(i, j)[1] = color[1];
                img.at<cv::Vec3b>(i, j)[2] = color[2];
            }
        }
        // cout << "Segments: " << segment_cnt << endl;
    }

    geometry::PointCloud pointcloud;
    auto pcd_ptr = make_shared<geometry::PointCloud>();
    if (!io::ReadPointCloud(file_name, pointcloud))
    {
        return 0;
    }

    *pcd_ptr = pointcloud;
    auto filtered_ptr = make_shared<geometry::PointCloud>();
    vector<vector<double>> base_z(height, vector<double>(width));
    vector<vector<double>> filtered_z(height, vector<double>(width));
    { // Down sampling
        vector<double> tans;
        double PI = acos(-1);
        double rad = (-16.6 - 0.265) * PI / 180;
        double delta_rad = 0.53 * PI / 180;
        double max_rad = (16.6 + 0.265) * PI / 180;
        while (rad < max_rad - 0.000001)
        {
            rad += delta_rad;
            tans.emplace_back(tan(rad));
        }

        for (int i = 0; i < pcd_ptr->points_.size(); i++)
        {
            double x = pcd_ptr->points_[i][1];
            double y = -pcd_ptr->points_[i][2];
            double z = -pcd_ptr->points_[i][0] + 0.03;
            pcd_ptr->points_[i] = Eigen::Vector3d(x, y, z);
            if (pcd_ptr->points_[i][2] > 0)
            {
                int u = (int)(width / 2 + f_x * x / z);
                int v = (int)(height / 2 + f_x * y / z);
                if (0 <= u && u < width && 0 <= v && v < height)
                {
                    auto it = lower_bound(tans.begin(), tans.end(), y / z);
                    int index = it - tans.begin();
                    if (index % 4 == 0)
                    {
                        if (img.at<cv::Vec3b>(v, u)[0] == 255 && img.at<cv::Vec3b>(v, u)[1] == 0 && img.at<cv::Vec3b>(v, u)[2] == 0)
                        {
                            continue;
                        }
                        else
                        {
                            filtered_ptr->points_.emplace_back(pcd_ptr->points_[i]);
                            filtered_ptr->colors_.emplace_back(img.at<cv::Vec3b>(v, u)[0] / 255.0, img.at<cv::Vec3b>(v, u)[1] / 255.0, img.at<cv::Vec3b>(v, u)[2] / 255.0);
                            img.at<cv::Vec3b>(v, u)[0] = 255;
                            img.at<cv::Vec3b>(v, u)[1] = 0;
                            img.at<cv::Vec3b>(v, u)[2] = 0;
                        }
                        filtered_z[v][u] = pcd_ptr->points_[i][2];
                    }
                    base_z[v][u] = pcd_ptr->points_[i][2];
                }
            }

            //pcd_ptr->points_[i][0] += 100;
        }
    }

    geometry::KDTreeSearchParamKNN kdtree_param(point_neighbors);
    { // Remove unstable points
        for (int i = 0; i < 3; i++)
        {
            if (filtered_ptr->points_.size() == 0)
            {
                break;
            }

            filtered_ptr->EstimateNormals(kdtree_param);
            vector<double> criterias;
            auto tree = make_shared<geometry::KDTreeFlann>(*filtered_ptr);
            for (int j = 0; j < filtered_ptr->points_.size(); j++)
            {
                vector<int> indexes(point_neighbors);
                vector<double> dists(point_neighbors);
                tree->SearchKNN(filtered_ptr->points_[j], point_neighbors, indexes, dists);
                double criteria = 0;
                for (int k = 0; k < indexes.size(); k++)
                {
                    int to = indexes[k];
                    criteria += 1;
                    for (int l = 0; l < 3; l++)
                    {
                        criteria -= abs(filtered_ptr->normals_[to][l] * filtered_ptr->normals_[i][l]);
                    }
                }
                criterias.emplace_back(criteria);
            }
            double avg = accumulate(criterias.begin(), criterias.end(), 0.0) / criterias.size();
            double std = sqrt(accumulate(criterias.begin(), criterias.end(), 0.0, [](double sum, double val) { return sum + val * val; }) / criterias.size() - avg * avg);

            vector<size_t> indicies;
            for (int j = 0; j < criterias.size(); j++)
            {
                if (abs(criterias[j] - avg) <= 3 * std)
                {
                    indicies.emplace_back(j);
                }
            }
            filtered_ptr = filtered_ptr->SelectByIndex(indicies);
        }
    }

    vector<vector<vector<int>>> bound(height, vector<vector<int>>(width));
    map<int, Eigen::VectorXd> interpolation_params;
    { // Point segmentation and interpolation
        filtered_ptr->EstimateNormals(kdtree_param);
        auto graph = make_shared<Graph>(filtered_ptr, point_neighbors, color_rate);
        auto unionFind = graph->segmentate(point_segment_k, point_size_min);

        map<int, vector<int>> segments;
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

            vector<Eigen::Vector2i> uvs;
            for (int i = 0; i < value.size(); i++)
            {
                double x = filtered_ptr->points_[value[i]][0];
                double y = filtered_ptr->points_[value[i]][1];
                double z = filtered_ptr->points_[value[i]][2];
                int u = (int)(width / 2 + f_x * x / z);
                int v = (int)(height / 2 + f_x * y / z);
                uvs.emplace_back((int)round(u), (int)round(v));
            }

            vector<Eigen::Vector2i> convex_hull = get_convex_hull(uvs);
            int color = (int)(filtered_ptr->colors_[key][0] * 255);
            for (int i = 0; i < convex_hull.size(); i++)
            {
                int yPrev = convex_hull[(i - 1 + convex_hull.size()) % convex_hull.size()][1];
                int x1 = convex_hull[i][0];
                int y1 = convex_hull[i][1];
                int x2 = convex_hull[(i + 1) % convex_hull.size()][0];
                int y2 = convex_hull[(i + 1) % convex_hull.size()][1];

                if (y1 == y2 && y1 != yPrev)
                {
                    bound_img.at<cv::Vec3b>(y1, x1)[0] = color;
                    bound_img.at<cv::Vec3b>(y1, x1)[1] = color;
                    bound_img.at<cv::Vec3b>(y1, x1)[2] = color;
                    bound[y1][x1].emplace_back(key);
                }
                else if (y1 != y2)
                {
                    double x1d = x1;
                    double delta = (x2 - x1) / (y2 - y1);
                    bool ignore = (y1 > yPrev && y1 > y2) || (y1 < yPrev && y1 < y2);
                    if (y1 < y2)
                    {
                        while (y1 < y2)
                        {
                            if (ignore)
                            {
                                ignore = false;
                            }
                            else
                            {
                                int x1i = (int)round(x1d);
                                bound_img.at<cv::Vec3b>(y1, x1i)[0] = color;
                                bound_img.at<cv::Vec3b>(y1, x1i)[1] = color;
                                bound_img.at<cv::Vec3b>(y1, x1i)[2] = color;
                                bound[y1][x1i].emplace_back(key);
                            }
                            y1++;
                            x1d += delta;
                        }
                    }
                    else if (y1 > y2)
                    {
                        while (y1 > y2)
                        {
                            if (ignore)
                            {
                                ignore = false;
                            }
                            else
                            {
                                int x1i = (int)round(x1d);
                                bound_img.at<cv::Vec3b>(y1, x1i)[0] = color;
                                bound_img.at<cv::Vec3b>(y1, x1i)[1] = color;
                                bound_img.at<cv::Vec3b>(y1, x1i)[2] = color;
                                bound[y1][x1i].emplace_back(key);
                            }
                            y1--;
                            x1d -= delta;
                        }
                    }
                }
            }

            vector<double> xs, ys, zs;
            for (int i = 0; i < value.size(); i++)
            {
                xs.emplace_back(filtered_ptr->points_[value[i]][0]);
                ys.emplace_back(filtered_ptr->points_[value[i]][1]);
                zs.emplace_back(filtered_ptr->points_[value[i]][2]);
            }

            Eigen::VectorXd linear_param(3);
            Eigen::VectorXd quadra_param(6);
            linear_param << 0, 0, 0;
            quadra_param << 0, 0, 0, 0, 0, 0;
            misra1a_functor linear_functor(value.size(), &xs[0], &ys[0], &zs[0]);
            misra1a_functor2 quadra_functor(value.size(), &xs[0], &ys[0], &zs[0]);

            Eigen::NumericalDiff<misra1a_functor> linear_numDiff(linear_functor);
            Eigen::LevenbergMarquardt<Eigen::NumericalDiff<misra1a_functor>> linear_lm(linear_numDiff);
            linear_lm.minimize(linear_param);

            if (value.size() >= 6)
            {
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

    map<int, set<int>> pixel_surface_map;
    {
        cv::Mat range_img = cv::Mat::zeros(height, width, CV_8UC3);
        for (int i = 0; i < height; i++)
        {
            set<int> used;
            stack<int> stk;
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < bound[i][j].size(); k++)
                {
                    int val = bound[i][j][k];
                    if (used.find(val) == used.end())
                    {
                        stk.push(val);
                        used.emplace(val);
                    }
                    else
                    {
                        used.erase(val);
                        while (stk.size() > 0 && used.find(stk.top()) == used.end())
                        {
                            stk.pop();
                        }
                    }
                }
                if (stk.size() > 0)
                {
                    int val = (int)(filtered_ptr->colors_[stk.top()][0] * 255);
                    range_img.at<cv::Vec3b>(i, j)[0] = val % 256;
                    range_img.at<cv::Vec3b>(i, j)[1] = (val / 256) % 256;
                    range_img.at<cv::Vec3b>(i, j)[2] = (val / 256 / 256) % 256;

                    int root = color_segments->root(i * width + j);
                    pixel_surface_map[root].insert(stk.top());
                }
            }
        }
    }

    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    vector<vector<double>> interpolated_z(height, vector<double>(width));
    { // Interpolation
        cv::Mat interpolated_range_img = cv::Mat::zeros(height, width, CV_8UC3);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int root = color_segments->root(i * width + j);
                if (base_z[i][j] > 0 && pixel_surface_map.find(root) != pixel_surface_map.end())
                {
                    int val = (int)(filtered_ptr->colors_[*pixel_surface_map[root].begin()][0] * 255);
                    interpolated_range_img.at<cv::Vec3b>(i, j)[0] = val % 256;
                    interpolated_range_img.at<cv::Vec3b>(i, j)[1] = (val / 256) % 256;
                    interpolated_range_img.at<cv::Vec3b>(i, j)[2] = (val / 256 / 256) % 256;

                    auto params = interpolation_params[*pixel_surface_map[root].begin()];
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
                        z = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
                    }

                    x = coef_a * z;
                    y = coef_b * z;

                    if (z > 0 && z < 100)
                    {
                        double color = img.at<cv::Vec3b>(i, j)[0] / 255.0;
                        interpolated_ptr->points_.emplace_back(x, y, z);
                        interpolated_ptr->colors_.emplace_back(color, color, color);
                        interpolated_z[i][j] = z;
                    }
                    else
                    {
                        //cout << params << endl;
                        //cout << x << " " << y << " " << z << endl;
                    }
                }
            }
        }
    }

    double error_res = 0;
    { // Evaluation
        double error = 0;
        int cnt = 0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (base_z[i][j] > 0 && filtered_z[i][j] == 0 && interpolated_z[i][j] > 0)
                {
                    error += abs((base_z[i][j] - interpolated_z[i][j]) / base_z[i][j]);
                    cnt++;
                }
            }
        }
        cout << "count=" << cnt << endl;
        error_res = error / cnt;
    }

    auto end = chrono::system_clock::now();
    double time = (double)(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000);
    cout << "Time[ms] = " << time << endl;

    /*
    Eigen::MatrixXd front(4, 4);
    front << 1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1;
    pcd_ptr->Transform(front);
    filtered_ptr->Transform(front);
    interpolated_ptr->Transform(front);

    cv::imshow("image", img);
    cv::imwrite("emoi.png", img);
    cv::waitKey(0);

    visualization::DrawGeometries({pcd_ptr, interpolated_ptr}, "PointCloud", 1600, 900);
*/

    return error_res;
}

int main(int argc, char *argv[])
{
    // Best
    cout << segmentation(28, 15, 0.5, 6, 0.9, 3, 0.5) << endl;
    //cout << segmentation(28, 15, 0.5, 6, 2.8, 3, 0) << endl;

    /*
    double best_error = 100;
    double best_color_segment_k = 40;
    int best_color_size_min = 3;
    for (double color_segment_k = 10; color_segment_k < 100; color_segment_k += 1)
    {
        for (int color_size_min = 1; color_size_min < 20; color_size_min += 1)
        {
            double error = segmentation(color_segment_k, color_size_min, 0.5, 6, 2.8, 3, 2.1);
            if (best_error > error)
            {
                best_error = error;
                best_color_segment_k = color_segment_k;
                best_color_size_min = color_size_min;
                cout << "Error = " << error << endl;
            }
        }
    }

    cout << "Color segment k = " << best_color_segment_k << endl;
    cout << "Color segment min = " << best_color_size_min << endl;
    */

    /*
    double best_error = 100;
    double best_rate = 0;
    double best_point_segment_k = 0;
    for (double color_rate = 0; color_rate < 10; color_rate += 0.1)
    {
        for (double point_segment_k = 0; point_segment_k < 3; point_segment_k += 0.1)
        {
            double error = segmentation(28, 15, 0.5, 6, point_segment_k, 3, color_rate);
            if (best_error > error)
            {
                best_error = error;
                best_rate = color_rate;
                best_point_segment_k = point_segment_k;
                cout << "Error = " << error << endl;
            }
        }
    }

    cout << "Color rate = " << best_rate << endl;
    cout << "Point segment k = " << best_point_segment_k << endl;
    */

    return 0;
}