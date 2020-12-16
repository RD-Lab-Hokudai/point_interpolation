#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"
#include "linear.cpp"

using namespace std;

struct UnionFind
{
    int d[1000 * 1000];
    UnionFind(int n = 0)
    {
        for (int i = 0; i < n; i++)
            d[i] = -1;
    }
    int root(int x)
    {
        if (d[x] < 0)
            return x;
        return d[x] = root(d[x]);
    }
    bool unite(int x, int y)
    {
        x = root(x);
        y = root(y);
        if (x == y)
            return false;
        if (d[x] > d[y])
            swap(x, y);
        d[x] += d[y];
        d[y] = x;
        return true;
    }
    bool same(int x, int y) { return root(x) == root(y); }
    int size(int x) { return -d[root(x)]; }
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

    shared_ptr<UnionFind> segmentate(double k)
    {
        auto startTime = chrono::system_clock::now();
        auto unionFind = make_shared<UnionFind>(length);
        int edge_len = edges.size();
        double *thresholds = new double[length];
        for (int i = 0; i < length; i++)
        {
            thresholds[i] = get_threshold(k, 1);
        }

        double diff_max = 0;
        double diff_min = 1000000;
        for (int i = 0; i < edge_len; i++)
        {
            double diff = get<0>(edges[i]);
            diff_max = max(diff_max, diff);
            diff_min = min(diff_min, diff);
        }
        int bucket_len = length;
        vector<int> *bucket = new vector<int>[bucket_len + 1];
        for (int i = 0; i < edge_len; i++)
        {
            int diff_level = (int)(bucket_len * (get<0>(edges[i]) - diff_min) / (diff_max - diff_min));
            bucket[diff_level].emplace_back(i);
        }
        cout << "Sort:" << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - startTime).count() << "ms" << endl;

        for (int i = 0; i <= bucket_len; i++)
        {
            for (int j = 0; j < bucket[i].size(); j++)
            {
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
        delete[] thresholds;
        delete[] bucket;

        /*
        sort(edges.begin(), edges.end());
        cout << "Sort:" << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - startTime).count() << "ms" << endl;
        for (int i = 0; i < edge_len; i++)
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
        */

        //cout << "Segmentation end" << endl;
        //cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - startTime).count() << "ms" << endl;

        return unionFind;
    }
};

int rate = 2;
vector<vector<double>> coefs;
void mouse_callback(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        cout << "clicked"
             << " " << y / rate << " " << x / rate << endl;
        cout << coefs[y / rate][x / rate] << endl;
    }
}

void original(vector<vector<double>> &target_grid, vector<vector<double>> &base_grid, vector<vector<int>> &target_vs, vector<vector<int>> &base_vs, EnvParams envParams, cv::Mat img, double color_segment_k, double sigma_s, double sigma_r, int r, double coef_s)
{
    vector<vector<double>> full_grid(envParams.height, vector<double>(envParams.width, 0));
    for (int i = 0; i < base_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            full_grid[base_vs[i][j]][j] = base_grid[i][j];
        }
    }

    // Original
    auto start = chrono::system_clock::now();
    shared_ptr<UnionFind> color_segments;
    {
        Graph graph(&img);
        cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;
        color_segments = graph.segmentate(color_segment_k);
    }
    cout << "Segmentation" << endl;
    cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;

    {
        cv::Mat seg_img = cv::Mat::zeros(envParams.height, envParams.width, CV_8UC3);
        random_device rnd;
        mt19937 mt(rnd());
        uniform_int_distribution<> rand(0, 255);
        for (int i = 0; i < envParams.height; i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                seg_img.at<cv::Vec3b>(i, j) = cv::Vec3b(rand(mt), rand(mt), rand(mt));
            }
        }
        for (int i = 0; i < envParams.height; i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                int root = color_segments->root(i * envParams.width + j);
                seg_img.at<cv::Vec3b>(i, j) = seg_img.at<cv::Vec3b>(root / envParams.width, root % envParams.width);
            }
        }
        cv::imshow("C", seg_img);
        cv::waitKey();
    }

    target_grid = vector<vector<double>>(target_vs.size(), vector<double>(envParams.width, 0));
    vector<vector<double>> coef_grid(img.rows, vector<double>(img.cols, 0));
    {
        for (int i = 0; i < target_vs.size(); i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                double coef = 0;
                double val = 0;
                int v = target_vs[i][j];
                // すでに点があるならそれを使う
                if (full_grid[v][j] > 0)
                {
                    target_grid[i][j] = full_grid[v][j];
                    continue;
                }

                cv::Vec3b d0 = img.at<cv::Vec3b>(v, j);
                int r0 = color_segments->root(v * envParams.width + j);

                for (int ii = 0; ii < r; ii++)
                {
                    for (int jj = 0; jj < r; jj++)
                    {
                        int dy = ii - r / 2;
                        int dx = jj - r / 2;
                        if (i + dy < 0 || i + dy >= target_vs.size() || j + dx < 0 || j + dx >= envParams.width)
                        {
                            continue;
                        }

                        int v1 = target_vs[i + dy][j + dx];
                        if (full_grid[v1][j + dx] <= 0)
                        {
                            continue;
                        }

                        cv::Vec3b d1 = img.at<cv::Vec3b>(v1, j + dx);
                        int r1 = color_segments->root(v1 * envParams.width + j + dx);
                        double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s) * exp(-cv::norm(d0 - d1) / 2 / sigma_r / sigma_r);
                        if (r1 != r0)
                        {
                            tmp *= coef_s;
                        }
                        val += tmp * full_grid[v1][j + dx];
                        coef += tmp;
                    }
                }
                if (coef > 0.3 /* some threshold */)
                {
                    target_grid[i][j] = val / coef;
                    coef_grid[target_vs[i][j]][j] = coef;
                }
            }
        }
    }

    {
        // Remove noise and Apply
        int dx[4] = {1, 1, 0, -1};
        int dy[4] = {0, 1, 1, 1};
        vector<vector<bool>> oks(target_vs.size(), vector<bool>(envParams.width, false));
        double rad_coef = 0.0005; //0.002
        for (int i = 0; i < target_vs.size(); i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                if (full_grid[i][j] > 0)
                {
                    // 元の点群はそのまま
                    continue;
                }
                if (target_vs[i][j] <= 0)
                {
                    continue;
                }

                double z = target_grid[i][j];
                double x = z * (j - envParams.width / 2) / envParams.f_xy;
                double y = z * (target_vs[i][j] - envParams.height / 2) / envParams.f_xy;

                double dist2 = x * x + y * y + z * z;
                double radius = rad_coef * dist2;

                bool ok = false;
                for (int k = 0; k < 4; k++)
                {
                    int ii = i + dy[k];
                    int jj = j + dx[k];
                    if (ii < 0 || ii >= target_vs.size() || jj < 0 || jj >= envParams.width)
                    {
                        continue;
                    }
                    if (target_grid[ii][jj] <= 0)
                    {
                        continue;
                    }

                    double z_tmp = target_grid[ii][jj];
                    double x_tmp = z * (jj - envParams.width / 2) / envParams.f_xy;
                    double y_tmp = z * (target_vs[ii][jj] - envParams.height / 2) / envParams.f_xy;
                    double distance2 = (x - x_tmp) * (x - x_tmp) + (y - y_tmp) * (y - y_tmp) + (z - z_tmp) * (z - z_tmp);
                    // ノイズ除去
                    if (distance2 <= radius * radius)
                    {
                        oks[i][j] = true;
                        oks[ii][jj] = true;
                    }
                }

                //oks[i][j] = true;

                if (!oks[i][j])
                {
                    target_grid[i][j] = 0;
                }
            }
        }
    }

    /*
    cv::Mat img2 = cv::Mat::zeros(img.rows * rate, img.cols * rate, CV_8UC3);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            for (int ii = 0; ii < rate; ii++)
            {
                for (int jj = 0; jj < rate; jj++)
                {
                    img2.at<cv::Vec3b>(i * rate + ii, j * rate + jj) = img.at<cv::Vec3b>(i, j);
                    if (coef_grid[i][j] > 0)
                    {
                        img2.at<cv::Vec3b>(i * rate + ii, j * rate + jj) = coef_grid[i][j] * cv::Vec3b(255, 255, 255);
                    }
                }
            }
        }
    }

    coefs = coef_grid;
    cv::imshow("win", img2);
    cv::setMouseCallback("win", mouse_callback);
    cv::waitKey();
    */

    cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;
}