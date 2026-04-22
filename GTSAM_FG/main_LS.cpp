#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

#include <iostream>
#include <fstream>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace gtsam;

// =========================================================================
// 1. FAKTOR DYNAMIKY (Linear Dynamics)
// =========================================================================
class LinearDynamicsFactor : public NoiseModelFactor2<Vector, Vector>
{
    Matrix F_;

public:
    LinearDynamicsFactor(Key key_prev, Key key_next, const Matrix &F, SharedNoiseModel model)
        : NoiseModelFactor2<Vector, Vector>(model, key_prev, key_next), F_(F) {}

    Vector evaluateError(const Vector &x_prev, const Vector &x_next,
                         Matrix *H_prev = nullptr, Matrix *H_next = nullptr) const override
    {
        Vector predicted = F_ * x_prev;

        if (H_prev)
            *H_prev = -F_;

        // OPRAVA: Dynamicky podle velikosti F (zde 3x3)
        if (H_next)
            *H_next = Matrix::Identity(F_.rows(), F_.cols());

        return x_next - predicted;
    }
};

// =========================================================================
// 2. FAKTOR MĚŘENÍ (Linear Measurement)
// =========================================================================
class LinearMeasurementFactor : public NoiseModelFactor1<Vector>
{
    Vector measured_;
    Matrix H_; // OPRAVA: Matici H si uložíme z konstruktoru

public:
    // OPRAVA: Konstruktor nyní přebírá matici H, aby rozměry seděly s main()
    LinearMeasurementFactor(Key key, const Vector &measured, const SharedNoiseModel &model, const Matrix &H_in)
        : NoiseModelFactor1<Vector>(model, key), measured_(measured), H_(H_in)
    {
    }

    Vector evaluateError(const Vector &x, Matrix *H = nullptr) const override
    {
        // x je 3x1, H_ je 2x3 -> vysledek 2x1 (odpovídá measured_)
        Vector h_x = H_ * x;

        if (H)
            *H = H_;

        return h_x - measured_;
    }
};

// Pomocná funkce pro simulaci dat
Vector sampleMultivariateNormal(const Matrix &Cov, std::mt19937 &gen)
{
    Eigen::LLT<Matrix> cholSolver(Cov);
    Matrix L = cholSolver.matrixL();
    Vector u(Cov.rows());
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < Cov.rows(); ++i)
        u(i) = dist(gen);
    return L * u;
}

int main()
{
    try
    {
        // -----------------------------------------------------------------------
        // A. DEFINICE MODELU (3 stavy, 2 měření)
        // -----------------------------------------------------------------------

        // F: 3x3
        Matrix F = Matrix::Identity(3, 3);
        F(0, 1) = 1;
        std::cout << "F=\n"
                  << F << std::endl;

        // H: 2x3 (Měříme 2 veličiny ze 3 stavů)
        Matrix H_sim(2, 3);
        H_sim << 1, 0, 1,
            1, 0, 0;
        std::cout << "H=\n"
                  << H_sim << std::endl;

        // Q: 3x3
        Matrix Q_mat(3, 3);
        Q_mat << 0.1, 0, 0,
            0, 0.1, 0,
            0, 0, 0.01;
        std::cout << "Q=\n"
                  << Q_mat << std::endl;

        // R: 2x2
        Matrix R_mat(2, 2);
        R_mat << 0.1, 0,
            0, 0.5;
        std::cout << "R=\n"
                  << R_mat << std::endl;

        // -----------------------------------------------------------------------
        // B. GENEROWÁNÍ DAT
        // -----------------------------------------------------------------------
        int N = 2000;
        std::vector<Vector> x_true_all(N + 1);
        std::vector<Vector> meas_all(N);

        std::mt19937 gen(42);

        // x0 má velikost 3
        Vector x0(3);
        x0 << 0, 0, 0;

        Matrix P0 = Matrix::Identity(3, 3);
        P0 = P0 * 0.001;

        x_true_all[0] = x0 + sampleMultivariateNormal(P0, gen);

        for (int k = 0; k < N; ++k)
        {
            Vector w_k = sampleMultivariateNormal(Q_mat, gen);
            Vector v_k = sampleMultivariateNormal(R_mat, gen);

            x_true_all[k + 1] = F * x_true_all[k] + w_k;
            meas_all[k] = H_sim * x_true_all[k] + v_k;
        }

        // -----------------------------------------------------------------------
        // C. SESTAVENÍ GTSAM GRAFU
        // -----------------------------------------------------------------------
        NonlinearFactorGraph graph;
        Values initialEstimate;

        auto procNoise = noiseModel::Gaussian::Covariance(Q_mat);
        auto measNoise = noiseModel::Gaussian::Covariance(R_mat);
        auto priorNoise = noiseModel::Gaussian::Covariance(P0);

        graph.add(PriorFactor<Vector>(Symbol('x', 0), x0, priorNoise));
        initialEstimate.insert(Symbol('x', 0), x0);

        for (int k = 0; k < N; ++k)
        {
            Symbol key_curr('x', k);
            Symbol key_next('x', k + 1);

            // OPRAVA: Předáváme H_sim do konstruktoru
            graph.add(LinearMeasurementFactor(key_curr, meas_all[k], measNoise, H_sim));
            graph.add(LinearDynamicsFactor(key_curr, key_next, F, procNoise));

            // Predikce pro počáteční odhad
            Vector prev_est = initialEstimate.at<Vector>(key_curr);
            Vector next_est_init = F * prev_est;
            initialEstimate.insert(key_next, next_est_init);
        }

        // -----------------------------------------------------------------------
        // D. OPTIMALIZACE
        // -----------------------------------------------------------------------
        gtsam::GaussNewtonParams params;
        params.maxIterations = 100;
        params.setVerbosity("ERROR");

        std::cout << "Spoustim optimalizaci..." << std::endl;
        gtsam::GaussNewtonOptimizer optimizer(graph, initialEstimate, params);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        Values result = optimizer.optimize();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

        std::cout << "Pocet iteraci: " << optimizer.iterations() << std::endl;
        std::cout << "Konecna chyba: " << optimizer.error() << std::endl;

        // =================================================================
        // E. VÝPOČET KOVARIANCÍ (Marginals)
        // =================================================================
        cout << "Počítám kovarianční matice..." << endl;
        gtsam::Marginals marginals(graph, result);

        ofstream outCov("full_covariance.csv");
        int dim = 3;

        for (size_t k = 0; k < N; ++k)
        {
            gtsam::Symbol key('x', k);
            gtsam::Matrix cov = marginals.marginalCovariance(key);

            outCov << k;
            for (int i = 0; i < dim; ++i)
            {
                for (int j = 0; j < dim; ++j)
                {
                    outCov << "," << cov(i, j);
                }
            }
            outCov << endl;
        }

        // -----------------------------------------------------------------------
        // F. VÝPIS VÝSLEDKŮ
        // -----------------------------------------------------------------------
        ofstream out("vysledky_linear.csv");
        // OPRAVA: Měření jsou jen 2, Stavy jsou jen 3. Odstraněny neplatné indexy.

        for (int k = 0; k <= N - 1; ++k)
        {
            Vector res = result.at<Vector>(Symbol('x', k));
            Vector truth = x_true_all[k];
            Vector meas = meas_all[k];

            out << k << ","
                << truth(0) << "," << truth(1) << "," << truth(2) << "," // True (3)
                << meas(0) << "," << meas(1) << ","                      // Meas (2)
                << res(0) << "," << res(1) << "," << res(2)              // Est (3)
                << endl;
        }

        std::cout << "Hotovo." << std::endl;
    }
    catch (std::exception &e)
    {
        std::cerr << "CHYBA: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}