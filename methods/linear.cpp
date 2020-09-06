#pragma once
#include <iostream>
#include <vector>

#include "../models/envParams.cpp"

using namespace std;

void linear(vector<vector<double>> &target_grid, vector<vector<double>> &base_grid, vector<vector<int>> &target_vs, vector<vector<int>> &base_vs, EnvParams envParams)
{
    // Linear interpolation
    vector<vector<double>> full_grid(envParams.height, vector<double>(envParams.width, 0));

    for (int j = 0; j < envParams.width; j++)
    {
        for (int k = 0; k < base_vs[0][j]; k++)
        {
            full_grid[k][j] = base_grid[0][j];
        }
    }

    // Need fix for acculate interpolation
    for (int i = 0; i + 1 < base_grid.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            double zPrev = base_grid[i][j];
            double zNext = base_grid[i + 1][j];
            double yPrev = base_grid[i][j] * (base_vs[i][j] - envParams.height / 2) / envParams.f_xy;
            double yNext = base_grid[i + 1][j] * (base_vs[i + 1][j] - envParams.height / 2) / envParams.f_xy;
            double angle = (zNext - zPrev) / (yNext - yPrev);

            for (int k = 0; base_vs[i][j] + k < base_vs[i + 1][j]; k++)
            {
                int v = base_vs[i][j] + k;
                double tan = (v - envParams.height / 2) / envParams.f_xy;
                double z = (zPrev - angle * yPrev) / (1 - angle * tan);
                full_grid[v][j] = z;
            }
        }
    }

    for (int j = 0; j < envParams.width; j++)
    {
        for (int k = 0; base_vs[(int)base_grid.size() - 1][j] + k < envParams.height; k++)
        {
            int v = base_vs[(int)base_grid.size() - 1][j] + k;
            full_grid[v][j] = base_grid[(int)base_grid.size() - 1][j];
        }
    }

    target_grid = vector<vector<double>>(target_vs.size(), vector<double>(envParams.width, 0));
    for (int i = 0; i < target_grid.size(); i++)
    {
        for (int j = 0; j < envParams.width; j++)
        {
            target_grid[i][j] = full_grid[target_vs[i][j]][j];
        }
    }
}