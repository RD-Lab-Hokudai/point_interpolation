#include <iostream>
#include <vector>
#include <stack>
#include <map>

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

    double get_diff(Eigen::Vector3d a, Eigen::Vector3d b)
    {
        double diff = 1;
        for (int i = 0; i < 3; i++)
        {
            diff -= abs(a[i] * b[i]);
        }
        return diff;
    }

    double get_threshold(int k, int size)
    {
        return 1.0 * k / size;
    }

public:
    Graph(shared_ptr<geometry::PointCloud> pcd_ptr, int neighbors)
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

                double diff = get_diff(pcd_ptr->normals_[i], pcd_ptr->normals_[to]);
                edges.emplace_back(diff, i, to);
            }
        }
    }

    shared_ptr<UnionFind> segmentate(int k)
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
            auto edge = edges[i];

            from = unionFind->root(from);
            to = unionFind->root(to);

            if (from == to)
            {
                continue;
            }

            if (diff <= thresholds[from] && diff <= thresholds[to])
            {
                unionFind->unite(from, to);
                int root = unionFind->root(from);
                thresholds[root] = diff + get_threshold(k, unionFind->size(root));
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

// A simplified version of examples/Cpp/Visualizer.cpp to demonstrate linking
// an external project to Open3D.
int main(int argc, char *argv[])
{
    clock_t start = clock();

    const int width = 640;
    const int height = 480;
    const double fx = 415.69219381653068;
    const double fy = 415.69219381653068;
    const double cx = 319.5;
    const double cy = 239.5;

    auto cloud_ptr = make_shared<geometry::PointCloud>();
    auto intrinsic_ptr = make_shared<camera::PinholeCameraIntrinsic>(width, height, fx, fy, cx, cy);
    auto rgb_ptr = make_shared<geometry::Image>();
    auto d_ptr = make_shared<geometry::Image>();
    if (!io::ReadImage("../../Open3D/examples/TestData/RGBD/color/00000.jpg", *rgb_ptr) ||
        !io::ReadImage("../../Open3D/examples/TestData/RGBD/depth/00000.png", *d_ptr))
    {
        return 1;
    }

    auto rgbd_ptr = geometry::RGBDImage::CreateFromColorAndDepth(*rgb_ptr, *d_ptr, 1000, 3, false);
    auto pcd_ptr = geometry::PointCloud::CreateFromRGBDImage(*rgbd_ptr, *intrinsic_ptr);
    Eigen::MatrixXd front(4, 4);
    front << 1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1;
    pcd_ptr->Transform(front);

    auto downed_ptr = pcd_ptr->VoxelDownSample(0.05);

    // Preprocess(remove unsustainable points)
    int neighbors = 6;
    int segment_k = 1; // in python code, 6
    auto filtered_ptr = downed_ptr;
    geometry::KDTreeSearchParamKNN kdtree_param(neighbors);
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
            vector<int> indexes(neighbors);
            vector<double> dists(neighbors);
            tree->SearchKNN(filtered_ptr->points_[j], neighbors, indexes, dists);
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

    filtered_ptr->EstimateNormals(kdtree_param);
    auto graph = make_shared<Graph>(filtered_ptr, neighbors);
    auto unionFind = graph->segmentate(segment_k);
    int segment_cnts = 0;

    vector<pair<int, int>> correspondences;
    for (int i = 0; i < filtered_ptr->points_.size(); i++)
    {
        int root = unionFind->root(i);
        if (i == root)
        {
            segment_cnts++;
        }
        correspondences.emplace_back(i, unionFind->root(i));
    }
    auto lineset_ptr = geometry::LineSet::CreateFromPointCloudCorrespondences(*filtered_ptr, *filtered_ptr, correspondences);

    map<int, vector<int>> segments;
    for (int i = 0; i < filtered_ptr->points_.size(); i++)
    {
        int root = unionFind->root(i);
        if (i == root)
        {
            segments[root] = {};
        }
        segments[root].emplace_back(i);
    }

    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat range = cv::Mat::zeros(height, width, CV_8UC3);
    vector<vector<vector<int>>> bound(height, vector<vector<int>>(width));
    map<int, Eigen::VectorXd> interpolation_params;
    for (const auto pair : segments)
    {
        auto key = pair.first;
        auto value = pair.second;
        if (value.size() < 6)
        {
            continue;
        }

        vector<Eigen::Vector2i> uvs;
        for (int i = 0; i < value.size(); i++)
        {
            double x = filtered_ptr->points_[value[i]][0];
            double y = filtered_ptr->points_[value[i]][1];
            double z = filtered_ptr->points_[value[i]][2];
            double u = (fx * x - cx * z) / (-z);
            double v = (-fy * y - cy * z) / (-z);
            uvs.emplace_back((int)round(u), (int)round(v));
        }

        int b = key % 256;
        int g = (key / 256) % 256;
        int r = (key / 65536) % 256;
        vector<Eigen::Vector2i> convex_hull = get_convex_hull(uvs);

        for (int i = 0; i < convex_hull.size(); i++)
        {
            int yPrev = convex_hull[(i - 1 + convex_hull.size()) % convex_hull.size()][1];
            int x1 = convex_hull[i][0];
            int y1 = convex_hull[i][1];
            int x2 = convex_hull[(i + 1) % convex_hull.size()][0];
            int y2 = convex_hull[(i + 1) % convex_hull.size()][1];

            if (y1 == y2 && y1 != yPrev)
            {
                image.at<cv::Vec3b>(y1, x1)[0] = b;
                image.at<cv::Vec3b>(y1, x1)[1] = g;
                image.at<cv::Vec3b>(y1, x1)[2] = r;
                bound[y1][x1].emplace_back(key);
            }
            else
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
                            image.at<cv::Vec3b>(y1, x1i)[0] = b;
                            image.at<cv::Vec3b>(y1, x1i)[1] = g;
                            image.at<cv::Vec3b>(y1, x1i)[2] = r;
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
                            image.at<cv::Vec3b>(y1, x1i)[0] = b;
                            image.at<cv::Vec3b>(y1, x1i)[1] = g;
                            image.at<cv::Vec3b>(y1, x1i)[2] = r;
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
        Eigen::NumericalDiff<misra1a_functor2> quadra_numDiff(quadra_functor);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<misra1a_functor>> linear_lm(linear_numDiff);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<misra1a_functor2>> quadra_lm(quadra_numDiff);
        linear_lm.minimize(linear_param);
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

    auto interpolated_ptr = make_shared<geometry::PointCloud>();
    vector<vector<Eigen::Vector3d>> poses(height, vector<Eigen::Vector3d>(width));
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
                int val = stk.top() + 1;
                range.at<cv::Vec3b>(i, j)[0] = val % 256;
                range.at<cv::Vec3b>(i, j)[1] = (val / 256) % 256;
                range.at<cv::Vec3b>(i, j)[2] = (val / 256 / 256) % 256;

                auto params = interpolation_params[val - 1];
                double coef_a = (cx - j) / fx;
                double coef_b = (i - cy) / fy;
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
                    z = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
                }

                x = coef_a * z;
                y = coef_b * z;

                interpolated_ptr->points_.emplace_back(Eigen::Vector3d(x, y, z));
                poses[i][j] << x, y, z;
            }
        }
    }

    auto original_ptr = make_shared<geometry::PointCloud>();
    double mre_interpolation = 0;
    for (int i = 0; i < height; i++)
    {
        float *p = (float *)(rgbd_ptr->depth_.data_.data() + i * rgbd_ptr->depth_.BytesPerLine());
        for (int j = 0; j < width; j++, p++)
        {
            double z = (double)(*p);
            if (z == 0)
            {
                continue;
            }
            z *= -1;
            double x = -(j - cx) * z / fx;
            double y = -(cy - i) * z / fy;
            original_ptr->points_.emplace_back(Eigen::Vector3d(x, y, z));

            //auto pcd_ptr = geometry::PointCloud::CreateFromRGBDImage(*rgbd_ptr, *intrinsic_ptr);

            int key = -1;
            key += range.at<cv::Vec3b>(i, j)[0];
            key += range.at<cv::Vec3b>(i, j)[1] * 256;
            key += range.at<cv::Vec3b>(i, j)[2] * 256 * 256;
            if (key == -1)
            {
                continue;
            }

            double inter_x = poses[i][j][0];
            double inter_y = poses[i][j][1];
            double inter_z = poses[i][j][2];
            double error = sqrt((inter_x - x) * (inter_x - x) + (inter_y - y) * (inter_y - y) + (inter_z - z) * (inter_z - z));
            if (isnan(error))
            {
                //cout << key << " " << inter_x << " " << inter_y << " " << inter_z << endl;
            }
            else
            {
                mre_interpolation += error;
            }
        }
    }
    //cout << mre_interpolation / interpolated_ptr->points_.size() << endl;

    clock_t end = clock();
    cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec." << endl;

    cv::imshow("hoge", range);
    cv::waitKey(0);
    cv::destroyWindow("hoge");
    visualization::DrawGeometries({/*pcd_ptr, interpolated_ptr*/ interpolated_ptr, original_ptr}, "PointCloud", 1600, 900);
    return 0;
}