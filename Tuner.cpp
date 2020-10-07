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
    EnvParams params_use = loadParams("miyanosawa_3_3_rgb_pwas_champ");
    HyperParams hyperParams = getDefaultHyperParams(params_use.isRGB);

    if (params_use.method == "pwas")
    {
        ofs = ofstream("pwas_tuning.csv", ios::app);
        double best_mre_sum = 1000000;
        double best_sigma_c = 1;
        double best_sigma_s = 1;
        double best_sigma_r = 1;
        int best_r = 1;
        // best params 2020/08/03 sigma_c:1000 sigma_s:1.99 sigma_r:19 r:7
        // best params 2020/08/10 sigma_c:12000 sigma_s:1.6 sigma_r:19 r:7
        // best params 2020/08/10 sigma_c:8000 sigma_s:1.6 sigma_r:19 r:7
        // 2020/9/18 12 0.5 10 1
        // 2020/10/6 10 0.7 91 7

        for (double sigma_c = 10; sigma_c <= 100; sigma_c += 10)
        {
            for (double sigma_s = 0.5; sigma_s < 2.5; sigma_s += 0.5)
            {
                for (double sigma_r = 1; sigma_r < 10; sigma_r += 1)
                {
                    for (int r = 7; r < 9; r += 2)
                    {
                        double mre_sum = 0;
                        hyperParams.pwas_sigma_c = sigma_c;
                        hyperParams.pwas_sigma_s = sigma_s;
                        hyperParams.pwas_sigma_r = sigma_r;
                        hyperParams.pwas_r = r;
                        for (int i = 0; i < params_use.data_ids.size(); i++)
                        {
                            double time, ssim, mse, mre;
                            interpolate(params_use.data_ids[i], params_use, hyperParams, time, ssim, mse, mre, false, false);
                            mre_sum += mre;
                        }

                        if (best_mre_sum > mre_sum)
                        {
                            best_mre_sum = mre_sum;
                            best_sigma_c = sigma_c;
                            best_sigma_s = sigma_s;
                            best_sigma_r = sigma_r;
                            best_r = r;
                            cout << "Updated : " << mre_sum / params_use.data_ids.size() << endl;
                            ofs << mre_sum / params_use.data_ids.size() << "," << sigma_c << "," << sigma_s << "," << sigma_r << "," << r << endl;
                        }
                    }
                }
            }
        }

        cout << "Sigma C = " << best_sigma_c << endl;
        cout << "Sigma S = " << best_sigma_s << endl;
        cout << "Sigma R = " << best_sigma_r << endl;
        cout << "R = " << best_r << endl;
        cout << "Mean error = " << best_mre_sum / params_use.data_ids.size() << endl;
    }
    if (params_use.method == "original")
    {
        ofs = ofstream("original_tuning.csv", ios::app);
        double best_mre_sum = 1000000;
        double best_color_segment_k = 1;
        double best_sigma_s = 1;
        double best_sigma_r = 1;
        int best_r = 1;
        double best_coef_s = 1;
        //110, 1.6, 19, 7, 0.7

        for (double color_segment_k = 110; color_segment_k <= 110; color_segment_k += 10)
        {
            for (double sigma_s = 2; sigma_s <= 2; sigma_s += 0.1)
            {
                for (double sigma_r = 16; sigma_r <= 18; sigma_r += 1)
                {
                    for (int r = 7; r < 9; r += 2)
                    {
                        for (double coef_s = 0; coef_s < 2; coef_s += 0.1)
                        {
                            double mre_sum = 0;
                            hyperParams.original_color_segment_k = color_segment_k;
                            hyperParams.original_sigma_s = sigma_s;
                            hyperParams.original_sigma_r = sigma_r;
                            hyperParams.original_r = r;
                            hyperParams.original_coef_s = coef_s;
                            for (int i = 0; i < params_use.data_ids.size(); i++)
                            {
                                double time, ssim, mse, mre;
                                interpolate(params_use.data_ids[i], params_use, hyperParams, time, ssim, mse, mre, false, false);
                                mre_sum += mre;
                            }

                            if (best_mre_sum > mre_sum)
                            {
                                best_mre_sum = mre_sum;
                                best_color_segment_k = color_segment_k;
                                best_sigma_s = sigma_s;
                                best_sigma_r = sigma_r;
                                best_r = r;
                                best_coef_s = coef_s;
                                cout << "Updated : " << mre_sum / params_use.data_ids.size() << endl;
                                ofs << mre_sum / params_use.data_ids.size() << "," << color_segment_k << "," << sigma_s << "," << sigma_r << "," << r << "," << coef_s << endl;
                            }
                        }
                    }
                }
            }
        }

        cout << "Sigma C = " << best_color_segment_k << endl;
        cout << "Sigma S = " << best_sigma_s << endl;
        cout << "Sigma R = " << best_sigma_r << endl;
        cout << "R = " << best_r << endl;
        cout << "Coef S = " << best_coef_s << endl;
        cout << "Mean error = " << best_mre_sum / params_use.data_ids.size() << endl;
    }
}