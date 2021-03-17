#pragma once
#include <map>

#include "../models/env_params.cpp"
#include "../models/hyper_params.cpp"

using namespace std;

EnvParams load_env_params(string params_name) {
  map<string, EnvParams> params;

  params["miyanosawa_20200303_rgb"] = {640, 480, 640, 506, 483,
                                       495, 568, 551, 510, false};
  params["miyanosawa_20200303_thermal"] = {
      938, 606, 938 / 2 * 1.01, 495, 466, 450, 469, 503, 487, false};

  params["miyanosawa_20200204_rgb"] = {640, 480, 640, 506, 483,
                                       495, 568, 551, 510, false};
  params["miyanosawa_20200204_thermal"] = {
      938, 606, 938 / 2 * 1.01, 495, 475, 458, 488, 568, 500, false};

  params["13jo_20200219_rgb"] = {672, 376, 672 / 2, 504, 474,
                                 493, 457, 489,     512, false};
  params["13jo_20200219_thermal"] = {
      938, 606, 938 / 2 * 1.01, 502, 484, 499, 478, 520, 502, false};

  params["hassamu_20201203_rgb"] = {640, 480, 640, 489, 492,
                                    510, 571, 529, 501, false};
  params["hassamu_20201203_thermal"] = {
      938, 606, 938 / 2 * 1.01, 486, 483, 448, 491, 472, 495, false};

  if (params.count(params_name)) {
    return params[params_name];
  } else {
    return params["miyanosawa_20200303_rgb"];
  }
}

HyperParams load_default_hyper_params() {
  HyperParams params;
  bool is_rgb = true;
  if (is_rgb) {
    params = {1.5, 1, 10, 1.6, 19, 7, 440, 1.3, 7, 0.32};
  } else {
    params = {1.5, 1, 10, 1.6, 19, 7, 440, 1.3, 7, 0.32};
    // Spare
    // params = {1.5, 1, 10, 1.6, 19, 7, 820, 1.6, 7, 0.03};
  }

  return params;
}