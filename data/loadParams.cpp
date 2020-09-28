#pragma once
#include <vector>
#include <map>

#include "../models/envParams.cpp"

using namespace std;

EnvParams loadParams(string params_name)
{
    vector<int> data_nos;
    for (int i = 1100; i <= 1300; i++)
    {
        data_nos.emplace_back(i);
    }

    vector<int> data_nos_13jo;
    for (int i = 10; i < 210; i++)
    {
        data_nos_13jo.emplace_back(i);
    }

    map<string, EnvParams> params;
    params["13jo"] = {938, 606, 938 / 2 * 1.01, 498, 485, 509, 481, 517, 500, "../../../data/2020_02_04_13jo/", {10, 20, 30, 40, 50}, "res_linear_13jo.csv", "linear", true, false};
    params["miyanosawa_linear_rgb_champ"] = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", {1207, 1262, 1264, 1265, 1277}, "res_linear_miyanosawa_RGB.csv", "linear", false, true};

    params["miyanosawa_3_3_rgb_linear"] = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_linear_miyanosawa_0303_1100-1300_RGB.csv", "linear", false, true};
    params["miyanosawa_3_3_rgb_pwas"] = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_pwas_miyanosawa_0303_1100-1300_RGB.csv", "pwas", false, true};
    params["miyanosawa_3_3_rgb_pwas_champ"] = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", {1207, 1262, 1264, 1265, 1277}, "res_pwas_miyanosawa_0303_RGB.csv", "pwas", false, true};
    params["miyanosawa_3_3_rgb_pwas_champ2"] = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", {100, 102, 104, 106, 108}, "res_pwas_miyanosawa_0303_RGB.csv", "pwas", false, true};
    params["miyanosawa_3_3_rgb_original"] = {640, 480, 640, 498, 489, 388, 554, 560, 506, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_original_miyanosawa_0303_1100-1300_RGB.csv", "original", false, true};

    params["miyanosawa_3_3_thermal_linear"] = {938, 606, 938 / 2 * 1.01, 495, 466, 450, 469, 503, 487, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_linear_miyanosawa_0303_1100-1300_Thermal.csv", "linear", false, false};
    params["miyanosawa_3_3_thermal_pwas"] = {938, 606, 938 / 2 * 1.01, 495, 466, 450, 469, 503, 487, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_pwas_miyanosawa_0303_1100-1300_Thermal.csv", "pwas", false, false};
    params["miyanosawa_3_3_thermal_original"] = {938, 606, 938 / 2 * 1.01, 495, 466, 450, 469, 503, 487, "../../../data/2020_03_03_miyanosawa/", data_nos, "res_original_miyanosawa_0303_1100-1300_Thermal.csv", "original", false, false};
    params["miyanosawa_3_3_thermal_original_champ"] = {938, 606, 938 / 2 * 1.01, 495, 466, 450, 469, 503, 487, "../../../data/2020_03_03_miyanosawa/", {1207, 1262, 1264, 1265, 1277}, "res_original_miyanosawa_0303_Thermal.csv", "original", false, false};

    params["miyanosawa_0204_rgb_linear"] = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_linear_miyanosawa_0204_1100-1300_RGB.csv", "linear", false, true};
    params["miyanosawa_0204_rgb_mrf"] = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_mrf_miyanosawa_0204_1100-1300_RGB.csv", "mrf", false, true};
    params["miyanosawa_0204_rgb_pwas"] = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_pwas_miyanosawa_0204_1100-1300_RGB.csv", "pwas", false, true};
    params["miyanosawa_0204_rgb_original"] = {640, 480, 640, 506, 483, 495, 568, 551, 510, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_original_miyanosawa_0204_1100-1300_RGB.csv", "original", false, true};

    params["miyanosawa_0204_thermal_linear"] = {938, 606, 938 / 2 * 1.01, 495, 475, 458, 488, 568, 500, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_linear_miyanosawa_0204_1100-1300_Thermal.csv", "linear", false, false};
    params["miyanosawa_0204_thermal_mrf"] = {938, 606, 938 / 2 * 1.01, 495, 475, 458, 488, 568, 500, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_mrf_miyanosawa_0204_1100-1300_Thermal.csv", "mrf", false, false};
    params["miyanosawa_0204_thermal_pwas"] = {938, 606, 938 / 2 * 1.01, 495, 475, 458, 488, 568, 500, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_pwas_miyanosawa_0204_1100-1300_Thermal.csv", "pwas", false, false};
    params["miyanosawa_0204_thermal_original"] = {938, 606, 938 / 2 * 1.01, 495, 475, 458, 488, 568, 500, "../../../data/2020_02_04_miyanosawa/", data_nos, "res_original_miyanosawa_0204_1100-1300_Thermal.csv", "original", false, false};

    params["13jo_0219_rgb_linear"] = {672, 376, 672 / 2, 504, 474, 493, 457, 489, 512, "../../../data/2020_02_19_13jo/", data_nos_13jo, "res_linear_13jo_0219_1100-1300_RGB.csv", "linear", false, true};
    params["13jo_0219_rgb_mrf"] = {672, 376, 672 / 2, 504, 474, 493, 457, 489, 512, "../../../data/2020_02_19_13jo/", data_nos_13jo, "res_mrf_13jo_0219_1100-1300_RGB.csv", "mrf", false, true};
    params["13jo_0219_rgb_pwas"] = {672, 376, 672 / 2, 504, 474, 493, 457, 489, 512, "../../../data/2020_02_19_13jo/", data_nos_13jo, "res_pwas_13jo_0219_1100-1300_RGB.csv", "pwas", false, true};
    params["13jo_0219_rgb_original"] = {672, 376, 672 / 2, 504, 474, 493, 457, 489, 512, "../../../data/2020_02_19_13jo/", data_nos_13jo, "res_original_13jo_0219_1100-1300_RGB.csv", "original", false, true};

    params["13jo_0219_thermal_linear"] = {938, 606, 938 / 2 * 1.01, 502, 484, 499, 478, 520, 502, "../../../data/2020_02_19_13jo/", data_nos_13jo, "res_linear_13jo_0219_1100-1300_Thermal.csv", "linear", false, false};
    params["13jo_0219_thermal_mrf"] = {938, 606, 938 / 2 * 1.01, 502, 484, 499, 478, 520, 502, "../../../data/2020_02_19_13jo/", data_nos_13jo, "res_mrf_13jo_0219_1100-1300_Thermal.csv", "mrf", false, false};
    params["13jo_0219_thermal_pwas"] = {938, 606, 938 / 2 * 1.01, 502, 484, 499, 478, 520, 502, "../../../data/2020_02_19_13jo/", data_nos_13jo, "res_pwas_13jo_0219_1100-1300_Thermal.csv", "pwas", true, false};
    params["13jo_0219_thermal_original"] = {938, 606, 938 / 2 * 1.01, 502, 484, 499, 478, 520, 502, "../../../data/2020_02_19_13jo/", data_nos_13jo, "res_original_13jo_0219_1100-1300_Thermal.csv", "original", false, false};

    if (params.count(params_name))
    {
        return params[params_name];
    }
    else
    {
        return params["miyanosawa_3_3_rgb_linear"];
    }
}