#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "utils.h"

using namespace std;

UnionFind::UnionFind(int n = 0) {
  for (int i = 0; i < n; i++) d[i] = -1;
}

int UnionFind::root(int x) {
  if (d[x] < 0) return x;
  return d[x] = root(d[x]);
}

bool UnionFind::unite(int x, int y) {
  x = root(x);
  y = root(y);
  if (x == y) return false;
  if (d[x] > d[y]) swap(x, y);
  d[x] += d[y];
  d[y] = x;
  return true;
}

bool UnionFind::same(int x, int y) { return root(x) == root(y); }

int UnionFind::size(int x) { return -d[root(x)]; }

double SegmentationGraph::get_diff(cv::Vec3b& a, cv::Vec3b& b) {
  double diff = 0;
  for (int i = 0; i < 3; i++) {
    diff += (a[i] - b[i]) * (a[i] - b[i]);
  }
  diff = sqrt(diff);
  return diff;
}

double SegmentationGraph::get_threshold(double k, int size) {
  return 1.0 * k / size;
}

SegmentationGraph::SegmentationGraph(cv::Mat* img) {
  length = img->rows * img->cols;
  int dx[] = {1, 0, 0, -1};
  int dy[] = {0, 1, -1, 0};
  for (int i = 0; i < img->rows; i++) {
    cv::Vec3b* row = img->ptr<cv::Vec3b>(i);
    for (int j = 0; j < img->cols; j++) {
      for (int k = 0; k < 2; k++) {
        int to_x = j + dx[k];
        int to_y = i + dy[k];
        if (0 <= to_x && to_x < img->cols && 0 <= to_y && to_y < img->rows) {
          double diff = get_diff(row[j], img->at<cv::Vec3b>(to_y, to_x));
          edges.emplace_back(diff, i * img->cols + j, to_y * img->cols + to_x);
        }
      }
    }
  }
}

shared_ptr<UnionFind> SegmentationGraph::segmentate(double k) {
  auto unionFind = make_shared<UnionFind>(length);
  int edge_len = edges.size();
  double* thresholds = new double[length];
  for (int i = 0; i < length; i++) {
    thresholds[i] = get_threshold(k, 1);
  }

  double diff_max = 0;
  double diff_min = 1000000;
  for (int i = 0; i < edge_len; i++) {
    double diff = get<0>(edges[i]);
    diff_max = max(diff_max, diff);
    diff_min = min(diff_min, diff);
  }
  int bucket_len = length;
  vector<int>* bucket = new vector<int>[bucket_len + 1];
  for (int i = 0; i < edge_len; i++) {
    int diff_level = (int)(bucket_len * (get<0>(edges[i]) - diff_min) /
                           (diff_max - diff_min));
    bucket[diff_level].emplace_back(i);
  }

  for (int i = 0; i <= bucket_len; i++) {
    for (int j = 0; j < bucket[i].size(); j++) {
      double diff = get<0>(edges[bucket[i][j]]);
      int from = get<1>(edges[bucket[i][j]]);
      int to = get<2>(edges[bucket[i][j]]);

      from = unionFind->root(from);
      to = unionFind->root(to);

      if (from == to) {
        continue;
      }

      if (diff <= min(thresholds[from], thresholds[to])) {
        unionFind->unite(from, to);
        int root = unionFind->root(from);
        thresholds[root] = diff + get_threshold(k, unionFind->size(root));
      }
    }
  }
  delete[] thresholds;
  delete[] bucket;

  // Quick sort
  /*
  sort(edges.begin(), edges.end());
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

  return unionFind;
}