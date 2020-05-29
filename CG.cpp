#include <iostream>
#include <vector>
#include <stack>
#include <map>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <time.h>

using namespace std;
using namespace Eigen;

int main(int argc, char *argv[])
{
    VectorXd z(9);
    z[4] = 10;
    SparseMatrix<double> W(9, 9);
    vector<Triplet<double>> W_list;
    W_list.emplace_back(4, 4, 3);
    W.setFromTriplets(W_list.begin(), W_list.end());

    SparseMatrix<double> S(9, 9);
    vector<Triplet<double>> S_list;
    int dire = 8;
    int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    //int dx[4] = {1, -1, 0, 0};
    //int dy[4] = {0, 0, 1, -1};
    int color[9] = {0, 0, 10, 0, 10, 10, 10, 10, 0};
    double c = 7;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            double wSum = 0;
            for (int k = 0; k < dire; k++)
            {
                int u = j + dx[k];
                int v = i + dy[k];
                if (0 <= u && u < 3 && 0 <= v && v < 3)
                {
                    double w = -sqrt(exp(-c * (color[i * 3 + j] - color[v * 3 + u]) * (color[i * 3 + j] - color[v * 3 + u])));
                    S_list.emplace_back(i * 3 + j, v * 3 + u, w);
                    wSum += w;
                }
            }
            S_list.emplace_back(i * 3 + j, i * 3 + j, -wSum);
        }
    }
    S.setFromTriplets(S_list.begin(), S_list.end());
    cout << W << endl;
    cout << "S=" << S << endl;
    SparseMatrix<double> A = W.transpose() * W + S.transpose() * S;
    cout << A << endl;
    VectorXd b = W.transpose() * W * z;
    cout << b << endl;

    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
    cg.compute(A);
    VectorXd x(9);
    //x << 0, 0, 4, 0, 4, 4, 4, 4, 0;
    //x = cg.solveWithGuess(b, x);
    x = cg.solve(b);
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error() << std::endl;
    cout << x << endl;
    cout << A * x << endl;
    return 0;
}