#include <iostream>

#include "models/envParams.cpp"
#include "models/hyperParams.cpp"
#include "data/loadParams.cpp"
#include "interpolate.cpp"

using namespace std;
using namespace open3d;

ofstream ofs;

int main(int argc, char *argv[])
{
    if (argc <= 3)
    {
        cout << "You should specify data folder, calibration setting name and interpolation method" << endl;
        return 1;
    }

    string data_folder_path = argv[1];
    DIR *dir;
    struct dirent *diread;
    set<string> file_names;
    if ((dir = opendir(data_folder_path.c_str())) != nullptr)
    {
        while ((diread = readdir(dir)) != nullptr)
        {
            file_names.insert(diread->d_name);
        }
        closedir(dir);
    }
    else
    {
        cout << "Invalid folder path!" << endl;
        return 1;
    }

    string params_name = argv[2];
    EnvParams params_use = load_env_params(params_name);
    HyperParams hyper_params= load_default_hyper_params();

    for (auto it = file_names.begin(); it != file_names.end(); it++)
    {
            string str = *it;

        try
        {
            string str = *it;
            size_t found = str.find(".png");
            if (found == string::npos)
            {
                throw 1;
            }

            string name = str.substr(0, found);
            string img_path = data_folder_path + name + ".png";
            cv::Mat img = cv::imread(img_path);

            string pcd_path = data_folder_path + name + ".pcd";
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) == -1)
            {
                throw 1;
            }

            for (int i = 0; i < cloud->points.size(); i++)
            {
                // Assign position for camera coordinates
                double x = cloud->points[i].y;
                double y = -cloud->points[i].z;
                double z = -cloud->points[i].x;

                cloud->points[i].x = x;
                cloud->points[i].y = y;
                cloud->points[i].z = z;
            }

            imgs.emplace_back(img);
            pcd_ptrs.emplace_back(cloud);

            height = img.rows;
            width = img.cols;
            f_x = width / 2;
            f_y = f_x;
            size_is_initialized = true;

        double time, ssim, mse, mre;
        interpolate(cloud,img, params_use, hyper_params, time, ssim, mse, mre, true);
        cout << str << "," << time << "," << ssim << "," << mse << "," << mre << "," << endl;
        }
        catch (int e)
        {
            cout<<"File ID "<<str<<": Either the image or the point cloud does not exist"<endl;
        }
    }
    return 0;
}