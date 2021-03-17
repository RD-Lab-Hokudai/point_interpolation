#pragma once

using namespace std;

struct EnvParams {
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

struct HyperParams {
  double mrf_k;
  double mrf_c;

  double pwas_sigma_c;
  double pwas_sigma_s;
  double pwas_sigma_r;
  int pwas_r;

  double original_color_segment_k;
  double original_sigma_s;
  int original_r;
  double original_coef_s;
};

EnvParams load_env_params(string params_name);

HyperParams load_default_hyper_params();