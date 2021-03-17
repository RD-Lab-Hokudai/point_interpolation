#include <iostream>

#include <dirent.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#include "interpolate.cpp"
#include "models.h"

using namespace std;

// Grid search
int main(int argc, char* argv[]) {
  if (argc < 4) {
    cout << "You must specify data folder, calibration setting name and "
            "interpolation method name"
         << endl;
    return 1;
  }

  string data_folder_path = argv[1];
  DIR* dir;
  struct dirent* diread;
  set<string> file_names;
  if ((dir = opendir(data_folder_path.c_str())) != nullptr) {
    while ((diread = readdir(dir)) != nullptr) {
      file_names.insert(diread->d_name);
    }
    closedir(dir);
  } else {
    cout << "Invalid folder path!" << endl;
    return 1;
  }

  string params_name = argv[2];
  EnvParams params_use = load_env_params(params_name);
  HyperParams hyper_params = load_default_hyper_params();

  string method_name = argv[3];
  if (!(method_name == "pwas" || method_name == "original")) {
    cout << "You must specify 'pwas' or 'original' as interpolation method name"
         << endl;
    return 1;
  }

  vector<cv::Mat> imgs;
  vector<pcl::PointCloud<pcl::PointXYZ>> clouds;
  for (auto it = file_names.begin(); it != file_names.end(); it++) {
    string str = *it;

    try {
      string str = *it;
      size_t found = str.find(".png");
      if (found == string::npos) {
        throw 1;
      }

      string name = str.substr(0, found);
      string img_path = data_folder_path + name + ".png";
      cv::Mat img = cv::imread(img_path);

      string pcd_path = data_folder_path + name + ".pcd";
      pcl::PointCloud<pcl::PointXYZ> cloud;
      if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, cloud) == -1) {
        throw 2;
      }

      for (int i = 0; i < cloud.points.size(); i++) {
        // Assign position for camera coordinates
        // Right-handed coordinate system
        double x = cloud.points[i].y;
        double y = -cloud.points[i].z;
        double z = -cloud.points[i].x;

        cloud.points[i].x = x;
        cloud.points[i].y = y;
        cloud.points[i].z = z;
      }

      imgs.push_back(img);
      clouds.push_back(cloud);
    } catch (int e) {
      switch (e) {
        case 1:
          break;
        case 2:
          cout << "Img " << str << ": The point cloud does not exist" << endl;
          break;
      }
    }
  }

  // 最大で20データまで使用
  int inc = imgs.size() >= 10 ? imgs.size() / 10 : 1;

  if (method_name == "pwas") {
    double best_mre_sum = 1000000;
    double best_sigma_c = 1;
    double best_sigma_s = 1;
    double best_sigma_r = 1;
    int best_r = 1;

    for (double sigma_c = 10; sigma_c <= 100; sigma_c += 10) {
      for (double sigma_s = 0.5; sigma_s <= 2.5; sigma_s += 0.5) {
        for (double sigma_r = 1; sigma_r <= 10; sigma_r += 1) {
          for (int r = 7; r <= 7; r += 1) {
            double mre_sum = 0;
            hyper_params.pwas_sigma_c = sigma_c;
            hyper_params.pwas_sigma_s = sigma_s;
            hyper_params.pwas_sigma_r = sigma_r;
            hyper_params.pwas_r = r;
            for (int i = 0; i < imgs.size(); i += inc) {
              double time, ssim, mse, mre, f_val;
              interpolate(clouds[i], imgs[i], params_use, hyper_params,
                          method_name, time, ssim, mse, mre, f_val, false);
              mre_sum += mre;
            }

            if (best_mre_sum > mre_sum) {
              best_mre_sum = mre_sum;
              best_sigma_c = sigma_c;
              best_sigma_s = sigma_s;
              best_sigma_r = sigma_r;
              best_r = r;
              cout << "Updated : " << mre_sum / imgs.size() << "," << sigma_c
                   << "," << sigma_s << "," << sigma_r << "," << r << endl;
            }
          }
        }
      }
    }

    cout << endl;
    cout << "Done." << endl;
    cout << "Mean error = " << best_mre_sum / imgs.size() << endl;
    cout << "Sigma C = " << best_sigma_c << endl;
    cout << "Sigma S = " << best_sigma_s << endl;
    cout << "Sigma R = " << best_sigma_r << endl;
    cout << "R = " << best_r << endl;
  }
  if (method_name == "original") {
    double best_mre_sum = 1000000;
    double best_color_segment_k = 1;
    double best_sigma_s = 1;
    int best_r = 1;
    double best_coef_s = 1;

    for (double color_segment_k = 400; color_segment_k <= 500;
         color_segment_k += 10) {
      for (double sigma_s = 1.6; sigma_s <= 1.6; sigma_s += 0.1) {
        for (int r = 7; r <= 7; r += 2) {
          for (double coef_s = 0.2; coef_s <= 0.4; coef_s += 0.01) {
            double mre_sum = 0;
            hyper_params.original_color_segment_k = color_segment_k;
            hyper_params.original_sigma_s = sigma_s;
            hyper_params.original_r = r;
            hyper_params.original_coef_s = coef_s;
            for (int i = 2; i < imgs.size(); i += inc) {
              double time, ssim, mse, mre, f_val;
              interpolate(clouds[i], imgs[i], params_use, hyper_params,
                          method_name, time, ssim, mse, mre, f_val, false);
              mre_sum += mre;
            }

            if (best_mre_sum > mre_sum) {
              best_mre_sum = mre_sum;
              best_color_segment_k = color_segment_k;
              best_sigma_s = sigma_s;
              best_r = r;
              best_coef_s = coef_s;
              cout << "Updated : " << mre_sum / imgs.size() << ","
                   << color_segment_k << "," << sigma_s << "," << r << ","
                   << best_coef_s << endl;
            }
          }
        }
      }
    }

    cout << endl;
    cout << "Done." << endl;
    cout << "Mean error = " << best_mre_sum / imgs.size() << endl;
    cout << "Sigma C = " << best_color_segment_k << endl;
    cout << "Sigma S = " << best_sigma_s << endl;
    cout << "R = " << best_r << endl;
    cout << "Coef S = " << best_coef_s << endl;
  }
}