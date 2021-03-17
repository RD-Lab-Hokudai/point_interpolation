#pragma once
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;

struct UnionFind {
  int d[1000 * 1000];
  UnionFind(int n);
  int root(int x);
  bool unite(int x, int y);
  bool same(int x, int y);
  int size(int x);
};

class SegmentationGraph {
  vector<tuple<double, int, int>> edges;
  int length;

  double get_diff(cv::Vec3b& a, cv::Vec3b& b);

  double get_threshold(double k, int size);

 public:
  SegmentationGraph(cv::Mat* img);

  shared_ptr<UnionFind> segmentate(double k);
};
