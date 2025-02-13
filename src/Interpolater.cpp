#include <iostream>

#include <dirent.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#include "interpolate.cpp"
#include "models.h"

using namespace std;

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

      double time, ssim, mse, mre, f_val;
      interpolate(cloud, img, params_use, hyper_params, method_name, time, ssim,
                  mse, mre, f_val, true);

      cout << name << "," << time << "," << ssim << "," << mse << "," << mre
           << "," << f_val << endl;

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
  return 0;
}