#include <Eigen/Core>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

#include "methods.h"
#include "models.h"
#include "utils.h"

using namespace std;

void linear(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
            EnvParams env_params) {
  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);

  // Horizontal interpolation
  for (int i = 0; i < vs.rows; i++) {
    double* row = src_grid.ptr<double>(i);
    double* dst_row = dst_grid.ptr<double>(i);

    // Left
    for (int j = 0; j < vs.cols; j++) {
      if (row[j] <= 1e-9) {
        continue;
      }
      for (int jj = 0; jj <= j; jj++) {
        dst_row[jj] = row[j];
      }
      break;
    }

    // Right
    for (int j = vs.cols - 1; j >= 0; j--) {
      if (row[j] <= 1e-9) {
        continue;
      }

      for (int jj = j; jj < vs.cols; jj++) {
        dst_row[jj] = row[j];
      }
      break;
    }

    int prev_u = 0;
    for (int j = 1; j < vs.cols; j++) {
      if (dst_row[j] <= 1e-9) {
        continue;
      }

      double prev_z = dst_row[prev_u];
      double prev_x =
          prev_z * (prev_u - env_params.width / 2) / env_params.f_xy;
      double next_z = dst_row[j];
      double next_x = next_z * (j - env_params.width / 2) / env_params.f_xy;
      for (int jj = prev_u; jj <= j; jj++) {
        double angle = (next_z - prev_z) / (next_x - prev_x);
        double tan = (jj - env_params.width / 2) / env_params.f_xy;
        double z = (prev_z - angle * prev_x) / (1 - tan * angle);
        dst_row[jj] = z;
      }
      prev_u = j;
    }
  }

  // Vertical interpolation
  for (int j = 0; j < vs.cols; j++) {
    // Up
    for (int i = 0; i < vs.rows; i++) {
      double now = dst_grid.at<double>(i, j);
      if (now <= 1e-9) {
        continue;
      }
      for (int ii = 0; ii <= i; ii++) {
        dst_grid.at<double>(ii, j) = now;
      }
      break;
    }

    // Down
    for (int i = vs.rows - 1; i >= 0; i--) {
      double now = dst_grid.at<double>(i, j);
      if (now <= 1e-9) {
        continue;
      }

      for (int ii = i; ii < vs.rows; ii++) {
        dst_grid.at<double>(ii, j) = now;
      }
      break;
    }

    int prev_i = 0;
    for (int i = 1; i < vs.rows; i++) {
      double now = dst_grid.at<double>(i, j);
      if (now <= 1e-9) {
        continue;
      }

      ushort prev_v = vs.at<ushort>(prev_i, j);
      ushort next_v = vs.at<ushort>(i, j);
      if (prev_v >= next_v) {
        continue;
      }

      double prev_z = dst_grid.at<double>(prev_i, j);
      double prev_y =
          prev_z * (prev_v - env_params.height / 2) / env_params.f_xy;
      double next_z = now;
      double next_y =
          next_z * (next_v - env_params.height / 2) / env_params.f_xy;
      for (int ii = prev_i; ii <= i; ii++) {
        ushort now_v = vs.at<ushort>(ii, j);
        double angle = (next_z - prev_z) / (next_y - prev_y);
        double tan = (now_v - env_params.height / 2) / env_params.f_xy;
        double z = (prev_z - angle * prev_y) / (1 - tan * angle);
        dst_grid.at<double>(ii, j) = z;
      }
      prev_i = i;
    }
  }
}

cv::Mat generateDiamondKernel(int kernel_size) {
  cv::Mat kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_8UC1);
  int center = kernel_size / 2;
  kernel.forEach<uchar>(
      [center, kernel_size](uchar& now, const int position[]) -> void {
        int dist = abs(center - position[0]) + abs(center - position[1]);

        if (dist <= center) {
          now = 1;
        }
      });
  return kernel;
}

void ip_basic(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
              EnvParams env_params) {
  double max_dist = 500;
  cv::Mat inverted = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  inverted.forEach<double>(
      [&src_grid, &max_dist](double& now, const int position[]) -> void {
        double d = src_grid.at<double>(position[0], position[1]);
        if (d > 0) {
          now = max_dist - d;
        }
      });

  cv::Mat dilate_kernel = generateDiamondKernel(5);

  cv::Mat dilated;
  cv::dilate(inverted, dilated, dilate_kernel);

  cv::Mat closed1, filled1;
  cv::Mat close_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(dilated, closed1, cv::MORPH_CLOSE, close_kernel);
  cv::Mat full_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  cv::dilate(closed1, filled1, full_kernel);
  filled1.forEach<double>(
      [&closed1](double& now, const int position[]) -> void {
        double d = closed1.at<double>(position[0], position[1]);
        if (d > 0) {
          now = d;
        }
      });

  for (int j = 0; j < vs.cols; j++) {
    int top = vs.rows;
    for (int i = 0; i < vs.rows; i++) {
      double val = filled1.at<double>(i, j);
      if (val > 0) {
        top = i;
        break;
      }
    }
    if (top == vs.rows) {
      continue;
    }

    double fill_val = filled1.at<double>(top, j);
    for (int i = 0; i < top; i++) {
      filled1.at<double>(i, j) = fill_val;
    }
  }
  cv::Mat filled2;
  cv::Mat full_fill_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31));
  cv::dilate(filled1, filled2, full_fill_kernel);
  filled2.forEach<double>(
      [&filled1](double& now, const int position[]) -> void {
        double d = filled1.at<double>(position[0], position[1]);
        if (d > 0) {
          now = d;
        }
      });

  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst_grid.forEach<double>(
      [&filled2, &max_dist](double& now, const int position[]) -> void {
        double d = filled2.at<double>(position[0], position[1]);
        if (d > 0) {
          now = max_dist - d;
        }
      });
}

void calc_exist_means(const cv::Mat& mat, cv::Mat& mean_mat, int& r,
                      const cv::Mat mask_mat) {
  cv::Mat mask_mean;
  cv::blur(mask_mat, mask_mean, cv::Size(r, r));

  cv::blur(mat, mean_mat, cv::Size(r, r));
  mean_mat.forEach<double>(
      [&mask_mean, &r](double& now, const int position[]) -> void {
        double mask = mask_mean.at<double>(position[0], position[1]);
        if (mask * r * r > 0.5) {
          now /= mask;
        } else {
          now = 0;
        }
      });
}

void guided_filter(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
                   EnvParams env_params, cv::Mat img) {
  cv::Mat gray = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  gray.forEach<double>(
      [&img, &vs, &src_grid](double& now, const int position[]) -> void {
        int v = vs.at<ushort>(position[0], position[1]);
        if (src_grid.at<double>(position[0], position[1]) <= 0) {
          return;
        }
        cv::Vec3b c = img.at<cv::Vec3b>(v, position[1]);
        now = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
        now /= 256;
      });

  int r = 11;
  double eps = 256 * 0.3;
  cv::Mat mask = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  mask.setTo(1, src_grid > 0);
  cv::Mat guide_mean, guide_square_mean, guide_variance;
  calc_exist_means(gray, guide_mean, r, mask);
  calc_exist_means(gray.mul(gray), guide_square_mean, r, mask);
  guide_variance = guide_square_mean - guide_mean.mul(guide_mean);
  cv::Mat depth_mean;
  cv::Mat guide_depth_mean, covariance;
  calc_exist_means(src_grid, depth_mean, r, mask);
  calc_exist_means(gray.mul(src_grid), guide_depth_mean, r, mask);
  covariance = guide_depth_mean - guide_mean.mul(depth_mean);
  cv::Mat a = covariance / (guide_variance + eps);
  cv::Mat b = depth_mean - a.mul(guide_mean);
  cv::Mat mean_a, mean_b;
  cv::blur(a, mean_a, cv::Size(r, r));
  cv::blur(b, mean_b, cv::Size(r, r));

  dst_grid = mean_a.mul(gray) + mean_b;
}

void mrf(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
         EnvParams env_params, cv::Mat img, double k, double c) {
  cv::Mat linear_grid;
  linear(src_grid, linear_grid, vs, env_params);

  int length = vs.rows * vs.cols;
  Eigen::VectorXd z_line(length);
  Eigen::SparseMatrix<double> W(length, length);
  vector<Eigen::Triplet<double>> W_triplets;
  for (int i = 0; i < vs.rows; i++) {
    for (int j = 0; j < vs.cols; j++) {
      double now = src_grid.at<double>(i, j);
      if (now > 0) {
        W_triplets.emplace_back(i * vs.cols + j, i * vs.cols + j, k);
      }
      z_line[i * vs.cols + j] = linear_grid.at<double>(i, j);
    }
  }
  W.setFromTriplets(W_triplets.begin(), W_triplets.end());

  Eigen::SparseMatrix<double> S(length, length);
  vector<Eigen::Triplet<double>> S_triplets;
  int dires = 4;
  int dx[4] = {1, -1, 0, 0};
  int dy[4] = {0, 0, 1, -1};
  for (int i = 0; i < vs.rows; i++) {
    for (int j = 0; j < vs.cols; j++) {
      double wSum = 0;
      ushort v0 = vs.at<ushort>(i, j);
      for (int k = 0; k < dires; k++) {
        int x = j + dx[k];
        int y = i + dy[k];
        if (0 <= x && x < vs.cols && 0 <= y && y < vs.rows) {
          int v1 = vs.at<double>(y, x);
          double x_norm2 =
              cv::norm(img.at<cv::Vec3b>(v0, j) - img.at<cv::Vec3b>(v1, x)) /
              (255 * 255);
          double w = -sqrt(exp(-c * x_norm2));
          S_triplets.emplace_back(i * vs.cols + j, y * vs.cols + x, w);
          wSum += w;
        }
      }
      S_triplets.emplace_back(i * vs.cols + j, i * vs.cols + j, -wSum);
    }
  }
  S.setFromTriplets(S_triplets.begin(), S_triplets.end());
  Eigen::SparseMatrix<double> A = S.transpose() * S + W.transpose() * W;
  Eigen::VectorXd b = W.transpose() * W * z_line;
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                           Eigen::Lower | Eigen::Upper>
      cg;
  cg.compute(A);
  Eigen::VectorXd y_res = cg.solve(b);

  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst_grid.forEach<double>(
      [&vs, &y_res](double& now, const int position[]) -> void {
        now = y_res[position[0] * vs.cols + position[1]];
      });
}

void pwas(const cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs, cv::Mat& img,
          double sigma_c, double sigma_s, double sigma_r, double r) {
  cv::Mat credibilities = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);

  int dx[] = {1, -1, 0, 0};
  int dy[] = {0, 0, 1, -1};
  credibilities.forEach<double>([&](double& now, const int position[]) -> void {
    cv::Vec3b val = 0;
    int cnt = 0;
    for (int k = 0; k < 4; k++) {
      int x = position[1] + dx[k];
      int y = vs.at<ushort>(position[0] + dy[k], x);
      if (x < 0 || x >= vs.cols || y < 0 || y >= vs.rows) {
        continue;
      }

      val += img.at<cv::Vec3b>(y, x);
      cnt++;
    }
    val -= cnt * img.at<cv::Vec3b>(vs.at<ushort>(position[0], position[1]),
                                   position[1]);
    now = exp(-cv::norm(val) / 2 / sigma_c / sigma_c);
  });

  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst_grid.forEach<double>([&](double& now, const int position[]) -> void {
    double coef = 0;
    double val = 0;

    // すでに点が与えられているならそれを使う
    double org_val = src_grid.at<double>(position[0], position[1]);
    if (now > 0) {
      now = org_val;
      return;
    }

    cv::Vec3b d0 =
        img.at<cv::Vec3b>(vs.at<ushort>(position[0], position[1]), position[1]);

    for (int ii = 0; ii < r; ii++) {
      for (int jj = 0; jj < r; jj++) {
        int dy = ii - r / 2;
        int dx = jj - r / 2;
        int tmp_y = position[0] + dy;
        int tmp_x = position[1] + dx;
        if (tmp_y < 0 || tmp_y >= vs.rows || tmp_x < 0 || tmp_x >= vs.cols) {
          continue;
        }

        if (src_grid.at<double>(tmp_y, tmp_x) <= 0) {
          continue;
        }

        cv::Vec3b d1 = img.at<cv::Vec3b>(vs.at<ushort>(tmp_y, tmp_x), tmp_x);
        double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s) *
                     exp(-cv::norm(d0 - d1) / 2 / sigma_r / sigma_r) *
                     credibilities.at<double>(tmp_y, tmp_x);
        val += tmp * src_grid.at<double>(tmp_y, tmp_x);
        coef += tmp;
      }
    }
    if (coef > 0) {
      now = val / coef;
    }
  });
}

void ext_jbu(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
             UnionFind& color_segments, EnvParams& vsenv_params,
             double color_segment_k, double sigma_s, int r, double coef_s) {
  dst_grid = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  dst_grid.forEach<double>([&](double& now, const int position[]) -> void {
    int y = position[0];
    int x = position[1];
    double coef = 0;
    double val = 0;

    // すでに点が与えられているならそれを使う
    double src_val = src_grid.at<double>(y, x);
    if (src_val > 0) {
      now = src_val;
      return;
    }

    int v = vs.at<ushort>(y, x);
    int r0 = color_segments.root(v * vsenv_params.width + x);

    for (int ii = 0; ii < r; ii++) {
      for (int jj = 0; jj < r; jj++) {
        int dy = ii - r / 2;
        int dx = jj - r / 2;
        if (y + dy < 0 || y + dy >= vs.rows || x + dx < 0 ||
            x + dx >= vs.cols) {
          continue;
        }

        double neighbor_val = src_grid.at<double>(y + dy, x + dx);
        if (neighbor_val <= 0) {
          continue;
        }

        int v1 = vs.at<ushort>(y + dy, x + dx);
        int r1 = color_segments.root(v1 * vsenv_params.width + x + dx);
        double tmp = exp(-(dx * dx + dy * dy) / 2 / sigma_s / sigma_s);
        if (r1 != r0) {
          tmp *= coef_s;
        }
        val += tmp * neighbor_val;
        coef += tmp;
      }
    }

    /* Bigger threshold will remove noises */
    if (coef > 1e-9) {
      now = val / coef;
    }
  });
}

void original(cv::Mat& src_grid, cv::Mat& dst_grid, cv::Mat& vs,
              EnvParams& env_params, cv::Mat& img, double color_segment_k,
              double sigma_s, int r, double coef_s) {
  shared_ptr<UnionFind> color_segments;
  SegmentationGraph graph(&img);
  color_segments = graph.segmentate(color_segment_k);
  ext_jbu(src_grid, dst_grid, vs, *color_segments, env_params, color_segment_k,
          sigma_s, r, coef_s);

  // 必要に応じて複数回実行
  /*
  cv::Mat twice_grid;
  ext_jbu(dst_grid, twice_grid, vs, *color_segments, env_params,
          color_segment_k, sigma_s, r, coef_s);
  dst_grid = twice_grid;
  */

  // 必要に応じてモルフォロジー処理
  /*
  double max_dist = 500;
  cv::Mat inverted = cv::Mat::zeros(vs.rows, vs.cols, CV_64FC1);
  inverted.forEach<double>(
      [&dst_grid, &max_dist](double& now, const int position[]) -> void {
        double d = dst_grid.at<double>(position[0], position[1]);
        if (d > 0) {
          now = max_dist - d;
        }
      });

  cv::Mat filled, filled2;
  cv::Mat fix_calibration_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  cv::morphologyEx(inverted, filled, cv::MORPH_CLOSE, fix_calibration_kernel);
  cv::Mat full_fill_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 31));
  cv::dilate(filled, filled2, full_fill_kernel);
  filled2.forEach<double>([&filled](double& now, const int position[]) -> void {
    double d = filled.at<double>(position[0], position[1]);
    if (d > 0) {
      now = d;
    }
  });

  dst_grid.forEach<double>(
      [&filled2, &max_dist](double& now, const int position[]) -> void {
        double d = filled2.at<double>(position[0], position[1]);
        if (d > 0) {
          now = max_dist - d;
        }
      });
  */
}