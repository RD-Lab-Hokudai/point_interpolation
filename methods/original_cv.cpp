#pragma once
#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "../models/envParams.cpp"
#include "../utils/UnionFind.cpp"
#include "../utils/SegmentationGraph.cpp"

using namespace std;

void ext_jbu(cv::Mat &src_mat, cv::Mat &dst_mat, cv::Mat &target_vs_mat, UnionFind &color_segments, EnvParams &envParams, double color_segment_k, double sigma_s, double sigma_r, int r, double coef_s)
{
    dst_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    dst_mat.forEach<double>([&](double &now, const int position[]) -> void {
        int y = position[0];
        int x = position[1];
        double coef = 0;
        double val = 0;
        // すでに点があるならそれを使う
        double src_val = src_mat.at<double>(y, x);
        if (src_val > 0)
        {
            now = src_val;
        }
        else
        {
            int v = target_vs_mat.at<int>(y, x);
            int r0 = color_segments.root(v * envParams.width + x);

            for (int ii = 0; ii < r; ii++)
            {
                for (int jj = 0; jj < r; jj++)
                {
                    int dy = ii - r / 2;
                    int dx = jj - r / 2;
                    if (y + dy < 0 || y + dy >= target_vs_mat.rows || x + dx < 0 || x + dx >= target_vs_mat.cols)
                    {
                        continue;
                    }

                    int v1 = target_vs_mat.at<int>(y + dy, x + dx);
                    double neighbor_val = src_mat.at<double>(y + dy, x + dx);
                    if (neighbor_val <= 0)
                    {
                        continue;
                    }

                    int r1 = color_segments.root(v1 * envParams.width + x + dx);
                    double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s);
                    if (r1 != r0)
                    {
                        tmp *= coef_s;
                    }
                    val += tmp * neighbor_val;
                    coef += tmp;
                }
            }
            if (coef > 0.0 /* some threshold */)
            {
                now = val / coef;
            }
        }
    });
}

void remove_noise(cv::Mat &src_mat, cv::Mat &dst_mat, cv::Mat &target_vs_mat, EnvParams envParams, double rad_coef = 0.001)
{
    int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    dst_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    dst_mat.forEach<double>([&](double &now, const int position[]) -> void {
        int y = position[0];
        int x = position[1];
        double src_val = src_mat.at<double>(y, x);
        if (src_val > 0)
        {
            now = src_val;
        }
        else
        {
            int v = target_vs_mat.at<int>(y, x);
            double z = src_mat.at<double>(y, x);
            double x = z * (x - envParams.width / 2) / envParams.f_xy;
            double y = z * (v - envParams.height / 2) / envParams.f_xy;

            double dist2 = x * x + y * y + z * z;
            double radius = rad_coef * dist2;

            int ok_cnt = 0;
            for (int k = 0; k < 8; k++)
            {
                int ii = y + dy[k];
                int jj = x + dx[k];
                if (ii < 0 || ii >= target_vs_mat.rows || jj < 0 || jj >= target_vs_mat.cols)
                {
                    continue;
                }

                int v_tmp = target_vs_mat.at<int>(ii, jj);
                double z_tmp = src_mat.at<double>(ii, jj);
                double x_tmp = z * (jj - envParams.width / 2) / envParams.f_xy;
                double y_tmp = z * (v_tmp - envParams.height / 2) / envParams.f_xy;
                double distance2 = (x - x_tmp) * (x - x_tmp) + (y - y_tmp) * (y - y_tmp) + (z - z_tmp) * (z - z_tmp);
                // ノイズ除去
                if (distance2 <= radius * radius)
                {
                    ok_cnt++;
                }
            }

            if (ok_cnt > 2)
            {
                now = z;
            }
        }
    });
}

void original_cv(cv::Mat &target_mat, cv::Mat &base_mat, cv::Mat &target_vs_mat, cv::Mat &base_vs_mat, EnvParams envParams, cv::Mat img, double color_segment_k, double sigma_s, double sigma_r, int r, double coef_s)
{
    cv::Mat full_mat = cv::Mat::zeros(envParams.height, envParams.width, CV_64FC1);
    base_mat.forEach<double>([&full_mat, &base_vs_mat](double &now, const int position[]) -> void {
        int v = base_vs_mat.at<int>(position[0], position[1]);
        full_mat.at<double>(v, position[1]) = now;
    });
    cv::Mat low_mat = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_64FC1);
    low_mat.forEach<double>([&full_mat, &target_vs_mat](double &now, const int position[]) -> void {
        int v = target_vs_mat.at<int>(position[0], position[1]);
        now = full_mat.at<double>(v, position[1]);
    });
    cv::imshow("A", low_mat);
    cv::waitKey();

    // Original
    auto start = chrono::system_clock::now();
    shared_ptr<UnionFind> color_segments;
    {
        Graph graph(&img);
        cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;
        color_segments = graph.segmentate(color_segment_k);
        /*
        cv::Mat hoge = cv::Mat::zeros(target_vs_mat.rows, target_vs_mat.cols, CV_8UC3);
        random_device rnd;
        mt19937 mt(rnd());
        uniform_int_distribution<> rand(0, 255);
        for (int i = 0; i < envParams.height; i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                int root = color_segments->root(i * envParams.width + j);
                hoge.at<cv::Vec3b>(i, j) = cv::Vec3b(rand(mt), rand(mt), rand(mt));
            }
        }
        for (int i = 0; i < envParams.height; i++)
        {
            for (int j = 0; j < envParams.width; j++)
            {
                int root = color_segments->root(i * envParams.width + j);
                hoge.at<cv::Vec3b>(i, j) = hoge.at<cv::Vec3b>(root / envParams.width, root % envParams.width);
            }
        }
        cv::imshow("C", hoge);
        cv::waitKey();
        */
    }
    cout << "Segmentation" << endl;
    cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;

    cv::Mat interpolated, noise_removed;
    ext_jbu(low_mat, interpolated, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(interpolated, noise_removed, target_vs_mat, envParams);
    /*
    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);
    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);

    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);
    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);
    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);
    ext_jbu(noise_removed, target_mat, target_vs_mat, *color_segments, envParams, color_segment_k, sigma_s, sigma_r, r, coef_s);
    remove_noise(target_mat, noise_removed, target_vs_mat, envParams);
*/
    target_mat = noise_removed;

    cout << chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - start).count() << "ms" << endl;
}