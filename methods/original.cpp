#pragma once
#include <iostream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"
#include "linear.cpp"

using namespace std;

struct UnionFind
{
    int d[1000 * 500];
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
        double *thresholds = new double[length];
        //vector<double> thresholds;
        double diff_max = 0;
        double diff_min = 1000000;
        for (int i = 0; i < length; i++)
        {
            thresholds[i] = get_threshold(k, 1);
            double diff = get<0>(edges[i]);
            diff_max = max(diff_max, diff);
            diff_min = min(diff_min, diff);
        }

        int bucket_len = 1000000;
        vector<int> *bucket = new vector<int>[bucket_len + 1];
        for (int i = 0; i < length; i++)
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
        cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - startTime).count() << "ms" << endl;
        sort(edges.begin(), edges.end());
        cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - startTime).count() << "ms" << endl;
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
        */

        cout << "Segmentation end" << endl;
        cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - startTime).count() << "ms" << endl;

        return unionFind;
    }
};

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
        cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;
    }
    cout << "Segmentation" << endl;
    cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;

    target_grid = vector<vector<double>>(target_vs.size(), vector<double>(envParams.width, 0));
    {
        for (int i = 0; i < target_vs.size(); i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                double coef = 0;
                double val = 0;
                int v = target_vs[i][j];
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
                if (coef > 0)
                {
                    target_grid[i][j] = val / coef;
                }
            }
        }
    }
    cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;
}