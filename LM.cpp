#include <iostream>

#include "Eigen/Dense"
#include "unsupported/Eigen/NonLinearOptimization"
#include "unsupported/Eigen/NumericalDiff"

using namespace Eigen;

// Generic functor
template <typename _Scalar, int NX = Dynamic, int NY = Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum
    {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;
};

struct misra1a_functor : Functor<double>
{
    misra1a_functor(int inputs, int values, double *x, double *y)
        : inputs_(inputs), values_(values), x(x), y(y) {}

    double *x;
    double *y;
    int operator()(const VectorXd &b, VectorXd &fvec) const
    {
        fvec[0] = 0;
        for (int i = 0; i < inputs_; ++i)
        {
            fvec[0] += (5 - b[i]) * (5 - b[i]);
        }
        //fvec[0] += (b[4] - 5) * (b[4] - 5);
        std::cout << fvec[0] << std::endl;
        return 0;
    }
    const int inputs_;
    const int values_;
    int inputs() const { return inputs_; }
    int values() const { return values_; }
};

int main(int argc, char *argv[])
{
    const int n = 9; // beta1とbeta2で二つ
    int info;

    VectorXd p(n); // beta1とbeta2の初期値(適当)
    for (int i = 0; i < n; i++)
    {
        p[i] = 4.9;
    }
    p[4] = 5;

    double xa[] = {77.6E0, 114.9E0, 141.1E0, 190.8E0, 239.9E0, 289.0E0, 332.8E0, 378.4E0, 434.8E0, 477.3E0, 536.8E0, 593.1E0, 689.1E0, 760.0E0};
    double ya[] = {10.07E0, 14.73E0, 17.94E0, 23.93E0, 29.61E0, 35.18E0, 40.02E0, 44.82E0, 50.76E0, 55.05E0, 61.01E0, 66.40E0, 75.47E0, 81.78E0};

    std::vector<double> x(&xa[0], &xa[14]); // vectorの初期化は不便
    std::vector<double> y(&ya[0], &ya[14]);

    misra1a_functor functor(n, x.size(), &x[0], &y[0]);

    NumericalDiff<misra1a_functor> numDiff(functor);
    LevenbergMarquardt<NumericalDiff<misra1a_functor>> lm(numDiff);
    info = lm.minimize(p);

    std::cout << p << std::endl; // 学習結果
}