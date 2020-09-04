#pragma once
#include <string>
#include <vector>

using namespace std;

struct EnvParams
{
    int width;
    int height;
    double f_xy;
    int X;
    int Y;
    int Z;
    int roll;
    int pitch;
    int yaw;

    string folder_path;
    vector<int> data_ids;
    string of_name;

    string method;

    bool isFullHeight;
    bool isRGB;
};