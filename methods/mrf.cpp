#pragma once
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include "../models/envParams.cpp"

using namespace std;

void mrf(vector<vector<double>> &target_grid, vector<vector<double>> &base_grid, vector<vector<int>> &target_vs, vector<vector<int>> &base_vs, EnvParams envParams, cv::Mat img, double k, double c)
{
    vector<vector<double>> base_full_grid(envParams.height, vector<double>(envParams.width, 0));
    for (int i = 0; i < base_grid.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            base_full_grid[base_vs[i][j]][j] = base_grid[i][j];
        }
    }

    int length = target_vs.size() * envParams.width;
    Eigen::VectorXd z_line(length);
    Eigen::SparseMatrix<double> W(length, length);
    vector<Eigen::Triplet<double>> W_triplets;
    for (int i = 0; i < target_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            if (base_full_grid[target_vs[i][j]][j] > 0)
            {
                z_line[i * envParams.width + j] = base_full_grid[target_vs[i][j]][j];
                W_triplets.emplace_back(i * envParams.width + j, i * envParams.width + j, k);
            }
        }
    }
    W.setFromTriplets(W_triplets.begin(), W_triplets.end());

    Eigen::SparseMatrix<double> S(length, length);
    vector<Eigen::Triplet<double>> S_triplets;
    int dires = 4;
    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};
    for (int i = 0; i < target_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            double wSum = 0;
            int v0 = target_vs[i][j];
            for (int k = 0; k < dires; k++)
            {
                int x = j + dx[k];
                int y = i + dy[k];
                if (0 <= x && x < envParams.width && 0 <= y && y < target_vs.size())
                {
                    int v1 = target_vs[y][x];
                    if (v1 == -1)
                    {
                        continue;
                    }
                    double x_norm2 = cv::norm(img.at<cv::Vec3b>(v0, j) - img.at<cv::Vec3b>(v1, x)) / 255 / 255;
                    double w = -sqrt(exp(-c * x_norm2));
                    S_triplets.emplace_back(i * envParams.width + j, y * envParams.width + x, w);
                    wSum += w;
                }
            }
            S_triplets.emplace_back(i * envParams.width + j, i * envParams.width + j, -wSum);
        }
    }
    S.setFromTriplets(S_triplets.begin(), S_triplets.end());
    Eigen::SparseMatrix<double> A = S.transpose() * S + W.transpose() * W;
    Eigen::VectorXd b = W.transpose() * W * z_line;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.compute(A);
    Eigen::VectorXd y_res = cg.solve(b);

    target_grid = vector<vector<double>>(target_vs.size(), vector<double>(envParams.width, 0));
    for (int i = 0; i < target_vs.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            target_grid[i][j] = y_res[i * envParams.width + j];
        }
    }
}