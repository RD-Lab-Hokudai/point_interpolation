#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"
#include "../utils/UnionFind.cpp"
#include "../utils/SegmentationGraph.cpp"
#include "linear.cpp"
#include "../preprocess/remove_noise.cpp"
#include "../postprocess/restore_pcd.cpp"

using namespace std;

int rate = 2;
vector<vector<double> > coefs;
void mouse_callback(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        cout << "clicked"
             << " " << y / rate << " " << x / rate << endl;
        cout << coefs[y / rate][x / rate] << endl;
    }
}

void original(vector<vector<double> > &target_grid, vector<vector<double> > &base_grid, vector<vector<int> > &target_vs, vector<vector<int> > &base_vs, EnvParams envParams, cv::Mat img, double color_segment_k, double sigma_s, double sigma_r, int r, double coef_s)
{
    vector<vector<double> > noise_removed;
    remove_noise(base_grid, noise_removed, base_vs, envParams);

    vector<vector<double> >
        full_grid(envParams.height, vector<double>(envParams.width, 0));
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
                int root = color_segments->root(i * envParams.width + j);
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

    target_grid = vector<vector<double> >(target_vs.size(), vector<double>(envParams.width, 0));
    vector<vector<double> > coef_grid(envParams.height, vector<double>(envParams.width, 0));
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
                        double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s); // * exp(-cv::norm(d0 - d1) / 2 / sigma_r / sigma_r);
                        if (r1 != r0)
                        {
                            tmp *= coef_s;
                        }
                        val += tmp * full_grid[v1][j + dx];
                        coef += tmp;
                    }
                }
                if (coef > 0 /*0.3*/ /* some threshold */)
                {
                    target_grid[i][j] = val / coef;
                    coef_grid[target_vs[i][j]][j] = coef;
                }
            }
        }
    }

    vector<vector<double> > last_noise_removed;
    remove_noise(target_grid, last_noise_removed, target_vs, envParams, 0.0001, 4);
    target_grid = last_noise_removed;
    auto ptr = shared_ptr<geometry::PointCloud>();
    restore_pcd_simple(target_grid, target_vs, envParams, ptr);
    visualization::DrawGeometries({ptr});

    cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;
}