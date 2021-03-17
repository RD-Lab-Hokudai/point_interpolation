#pragma once
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>

using namespace std;

// quality-metric
namespace qm {
#define C1 (float)(0.01 * 100 * 0.01 * 100)
#define C2 (float)(0.03 * 100 * 0.03 * 100)

// sigma on block_size
double sigma(cv::Mat& m, int i, int j, int block_size) {
  double sd = 0;

  cv::Mat m_tmp = m(cv::Range(i, i + block_size), cv::Range(j, j + block_size));
  cv::Mat m_squared(block_size, block_size, CV_64F);

  multiply(m_tmp, m_tmp, m_squared);

  // E(x)
  double avg = mean(m_tmp)[0];
  // E(x²)
  double avg_2 = mean(m_squared)[0];

  sd = sqrt(avg_2 - avg * avg);

  return sd;
}

// Covariance
double cov(cv::Mat& m1, cv::Mat& m2, int i, int j, int block_size) {
  cv::Mat m3 = cv::Mat::zeros(block_size, block_size, m1.depth());
  cv::Mat m1_tmp =
      m1(cv::Range(i, i + block_size), cv::Range(j, j + block_size));
  cv::Mat m2_tmp =
      m2(cv::Range(i, i + block_size), cv::Range(j, j + block_size));

  multiply(m1_tmp, m2_tmp, m3);

  double avg_ro = mean(m3)[0];     // E(XY)
  double avg_r = mean(m1_tmp)[0];  // E(X)
  double avg_o = mean(m2_tmp)[0];  // E(Y)

  double sd_ro = avg_ro - avg_o * avg_r;  // E(XY) - E(X)E(Y)

  return sd_ro;
}

// Mean reprojection error
double mre(cv::Mat& img1, cv::Mat& img2) {
  double error = 0;
  int height = img1.rows;
  int width = img1.cols;
  int cnt = 0;

  double o_min = 1000;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      double o = img1.at<double>(i, j);
      double r = img2.at<double>(i, j);
      if (o > 1e-9 && r > 1e-9) {
        error += abs((o - r) / o);
        cnt++;
        o_min = min(o_min, o);
      }
    }
  }

  if (cnt == 0) {
    return 1e9;
  } else {
    return error / cnt;
  }
}

// Mean squared error
double eqm(cv::Mat& img1, cv::Mat& img2) {
  double eqm = 0;
  int height = img1.rows;
  int width = img1.cols;
  int cnt = 0;
  int cannot = 0;
  int ground_cnt = 0;
  int interpolate_cnt = 0;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      double o = img1.at<double>(i, j);
      double r = img2.at<double>(i, j);
      if (o > 1e-9 && r > 1e-9) {
        eqm += (o - r) * (o - r);
        cnt++;
      }
      if (o > 1e-9 && r <= 1e-9) {
        cannot++;
      }
      if (o > 1e-9) {
        ground_cnt++;
      }
      if (r > 1e-9) {
        interpolate_cnt++;
      }
    }
  }

  if (cnt == 0) {
    return 1e9;
  } else {
    return eqm / cnt;
  }
}

/**
 *	Compute the PSNR between 2 images
 */
double psnr(cv::Mat& img_src, cv::Mat& img_compressed, int block_size) {
  int D = 255;
  return (10 * log10((D * D) / eqm(img_src, img_compressed)));
}

/**
 * Compute the SSIM between 2 images
 */
double ssim(cv::Mat& img1, cv::Mat& img2, int block_size) {
  double mssim = 0;

  int nbBlockPerHeight = img1.rows / block_size;
  int nbBlockPerWidth = img1.cols / block_size;
  int validBlocks = 0;

  for (int k = 0; k < nbBlockPerHeight; k++) {
    for (int l = 0; l < nbBlockPerWidth; l++) {
      int m = k * block_size;
      int n = l * block_size;

      int cnt = 0;
      double avg_o = 0;
      double avg_r = 0;
      double avg2_o = 0;
      double avg2_r = 0;
      double avg_or = 0;
      for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
          double o = img1.at<double>(m + i, n + j);
          double r = img2.at<double>(m + i, n + j);
          if (o > 1e-9 && r > 1e-9) {
            avg_o += o;
            avg2_o += o * o;
            avg_r += r;
            avg2_r += r * r;
            avg_or += o * r;
            cnt++;
          }
        }
      }

      if (cnt == 0) {
      } else {
        avg_o /= cnt;
        avg2_o /= cnt;
        avg_r /= cnt;
        avg2_r /= cnt;
        avg_or /= cnt;

        double sigma2_o = avg2_o - avg_o * avg_o;
        double sigma2_r = avg2_r - avg_r * avg_r;
        double sigma_or = avg_or - avg_o * avg_r;

        double ssim =
            ((2 * avg_o * avg_r + C1) * (2 * sigma_or + C2)) /
            ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma2_o + sigma2_r + C2));
        ssim = min(1.0, ssim);
        ssim = max(0.0, ssim);
        mssim += ssim;

        validBlocks++;
      }
    }
  }

  if (validBlocks == 0) {
    return 0;
  } else {
    return mssim / validBlocks;
  }
}

double f_value(cv::Mat& img1, cv::Mat& img2) {
  int tp = 0;
  int fp = 0;
  int fn = 0;
  int tn = 0;
  int height = img1.rows;
  int width = img1.cols;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      double o = img1.at<double>(i, j);
      double r = img2.at<double>(i, j);
      if (o > 1e-9 && r > 1e-9) {
        tp++;
      } else if (o > 1e-9) {
        fn++;
      } else if (r > 1e-9) {
        fp++;
      } else {
        tn++;
      }
    }
  }

  double precision = (0.0 + tp) / (tp + fp);
  double recall = (0.0 + tp) / (tp + fn);
  return 2 * precision * recall / (precision + recall);
}
}  // namespace qm