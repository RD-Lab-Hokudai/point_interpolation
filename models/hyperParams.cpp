#pragma once

using namespace std;

struct HyperParams
{
    double mrf_k;
    double mrf_c;

    double pwas_sigma_c;
    double pwas_sigma_s;
    double pwas_sigma_r;
    int pwas_r;

    double original_color_segment_k;
    double original_sigma_s;
    double original_sigma_r;
    int original_r;
    double original_coef_s;
};

HyperParams getDefaultHyperParams(bool is_rgb = true)
{
    HyperParams params;
    if (is_rgb)
    {
        params = {1.0, 1000, 10, 1.6, 19, 7, 110, 1.6, 19, 7, 0.7};
    }
    else
    {
        params = {1.0, 1000, 10, 1.6, 19, 7, 110, 1.6, 19, 7, 0.7};
    }

    return params;
}