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
    VectorXd x(2), b(2);
    SparseMatrix<double> A(2, 2);
    vector<Triplet<double>> list;
    list.emplace_back(Triplet<double>(0, 0, -2));
    list.emplace_back(Triplet<double>(0, 1, -1));
    list.emplace_back(Triplet<double>(1, 0, 1));
    list.emplace_back(Triplet<double>(1, 1, 2));
    A.setFromTriplets(list.begin(), list.end());
    b[0] = 3;
    b[1] = 0;
    SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    x = solver.solve(b);
    cout << x << endl;
    return 0;
}