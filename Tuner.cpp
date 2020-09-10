#include <iostream>

#include "models/envParams.cpp"
#include "data/loadParams.cpp"
#include "interpolate.cpp"

using namespace std;
using namespace open3d;

int main(int argc, char *argv[])
{
    EnvParams params_use = loadParams("miyanosawa_3_3_rgb_pwas_champ");

    if (params_use.method == "pwas")
    {
        double best_mre = 1000000;
        double best_sigma_c = 1;
        double best_sigma_s = 1;
        double best_sigma_r = 1;
        int best_r = 1;
        // best params 2020/08/03 sigma_c:1000 sigma_s:1.99 sigma_r:19 r:7
        // best params 2020/08/10 sigma_c:12000 sigma_s:1.6 sigma_r:19 r:7
        // best params 2020/08/10 sigma_c:8000 sigma_s:1.6 sigma_r:19 r:7

        for (double sigma_c = 10; sigma_c <= 1000; sigma_c += 10)
        {
            for (double sigma_s = 0.1; sigma_s < 1.7; sigma_s += 0.1)
            {
                for (double sigma_r = 1; sigma_r < 100; sigma_r += 10)
                {
                    for (int r = 1; r < 9; r += 2)
                    {
                        double error = 0;
                        for (int i = 0; i < params_use.data_ids.size(); i++)
                        {
                            //error += segmentate(params_use.data_ids[i], params_use);
                        }

                        if (best_error > error)
                        {
                            best_error = error;
                            best_sigma_c = sigma_c;
                            best_sigma_s = sigma_s;
                            best_sigma_r = sigma_r;
                            best_r = r;
                        }
                    }
                }
            }
        }

        cout << "Sigma C = " << best_sigma_c << endl;
        cout << "Sigma S = " << best_sigma_s << endl;
        cout << "Sigma R = " << best_sigma_r << endl;
        cout << "R = " << best_r << endl;
        cout << "Mean error = " << best_error / params_use.data_ids.size() << endl;
    }
}