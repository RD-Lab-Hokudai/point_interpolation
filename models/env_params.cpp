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

    bool isFullHeight;
};