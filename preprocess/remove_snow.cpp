#pragma once
#include <vector>

#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace open3d;

shared_ptr<geometry::PointCloud> remove_snow(shared_ptr<geometry::PointCloud> pcd_ptr)
{
    auto removed_ptr = make_shared<geometry::PointCloud>();
    auto kdtree = make_shared<geometry::KDTreeFlann>(*pcd_ptr);
    double rad_coef = 0.002;
    for (int i = 0; i < pcd_ptr->points_.size(); i++)
    {
        double x = pcd_ptr->points_[i][0];
        double y = pcd_ptr->points_[i][1];
        double z = pcd_ptr->points_[i][2];
        double distance = sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0));

        //探索半径：係数*(距離)^2
        double radius = rad_coef * pow(distance, 2.0);

        //半径r以内にある点を探索
        vector<int> indexes(2);
        vector<double> dists(2);
        kdtree->SearchKNN(pcd_ptr->points_[i], 2, indexes, dists);

        //radiusを超えない範囲に近傍点があれば残す
        if (dists[1] <= radius)
        {
            removed_ptr->points_.emplace_back(pcd_ptr->points_[i]);
        }
    }

    return removed_ptr;
}
