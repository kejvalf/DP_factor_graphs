#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>
#include <gtsam/nonlinear/Marginals.h> // PŘIDÁNO PRO VÝPOČET KOVARIANCE

// Pouze GTSAM hlavičky pro matematiku
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

#include "MapUtils.h"

#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

// =========================================================================
// 1. FAKTOR DYNAMIKY
// =========================================================================
class DynamicsFactor : public gtsam::NoiseModelFactor2<gtsam::Vector, gtsam::Vector>
{
    double dt_;

public:
    DynamicsFactor(gtsam::Key key_prev, gtsam::Key key_next, double dt, gtsam::SharedNoiseModel model)
        : gtsam::NoiseModelFactor2<gtsam::Vector, gtsam::Vector>(model, key_prev, key_next), dt_(dt) {}

    gtsam::Vector evaluateError(const gtsam::Vector &x_prev, const gtsam::Vector &x_next, gtsam::Matrix *H_prev = nullptr, gtsam::Matrix *H_next = nullptr) const override
    {
        gtsam::Vector predicted = x_prev;
        predicted(0) += dt_ * x_prev(2);
        predicted(1) += dt_ * x_prev(3);

        if (H_prev)
        {
            gtsam::Matrix H = -gtsam::Matrix::Identity(4, 4);
            H(0, 2) = -dt_;
            H(1, 3) = -dt_;
            *H_prev = H;
        }
        if (H_next)
        {
            *H_next = gtsam::Matrix::Identity(4, 4);
        }

        return x_next - predicted;
    }
};

// =========================================================================
// 2. FAKTOR MĚŘENÍ
// =========================================================================
class CustomMeasurementFactor : public gtsam::NoiseModelFactor1<gtsam::Vector>
{
    gtsam::Vector measured_;
    const MapInterpolant &map_;

public:
    CustomMeasurementFactor(gtsam::Key key, const gtsam::Vector &measured, const gtsam::SharedNoiseModel &model, const MapInterpolant &map)
        : gtsam::NoiseModelFactor1<gtsam::Vector>(model, key), measured_(measured), map_(map) {}

    gtsam::Vector evaluateError(const gtsam::Vector &x, gtsam::Matrix *H = nullptr) const override
    {

        double px = x(0);
        double py = x(1);
        double vx = x(2);
        double vy = x(3);

        double h_baro = map_.getZ(px, py);

        // double h_vx = vx;
        // double h_vy = vy;
        double h_v = sqrt(vx * vx + vy * vy);

        // gtsam::Vector h_x(3);
        // h_x << h_baro, h_vx, h_vy;

        gtsam::Vector h_x(2);
        h_x << h_baro, h_v;

        if (H)
        {
            // gtsam::Matrix J = gtsam::Matrix::Zero(3, 4);
            gtsam::Matrix J = gtsam::Matrix::Zero(2, 4);

            double dZdx, dZdy;
            map_.getGradients(px, py, dZdx, dZdy);

            J(0, 0) = dZdx;
            J(0, 1) = dZdy;

            // pro linearni nereni rychlosti
            // J(1, 2) = 1;
            // J(1, 3) = 0;
            // J(2, 2) = 0;
            // J(2, 3) = 1;

            // pro nelinearni nereni rychlosti
            J(1, 2) = vx / (sqrt(vx * vx + vy * vy));
            J(1, 3) = vy / (sqrt(vx * vx + vy * vy));

            *H = J;
        }
        return h_x - measured_;
    }
};

int main()
{
    try
    {
        // 1. NAČTENÍ SUROVÝCH DAT
        auto raw_x = flatten(readCSV("../mapX.csv"));
        auto raw_y = flatten(readCSV("../mapY.csv"));
        auto raw_z = readCSV("../mapZ.csv");

        auto hB_data = flatten(readCSV("../hB.csv"));
        std::vector<double> mx = flatten(readCSV("../GNSS_X.csv"));
        std::vector<double> my = flatten(readCSV("../GNSS_Y.csv"));

        size_t N = 200; // min(mx.size(), hB_data.size()); // 200
        if (N == 0)
            return 0;

        // =================================================================
        // KROK A: NORMALIZACE (POSUN DO LOKÁLNÍ NULY)
        // =================================================================
        double originX = mx[0];
        double originY = my[0];

        std::vector<double> mapX_norm = raw_x;
        std::vector<double> mapY_norm = raw_y;
        for (double &val : mapX_norm)
            val -= originX;
        for (double &val : mapY_norm)
            val -= originY;
        MapInterpolant map(mapX_norm, mapY_norm, raw_z);

        std::vector<double> mx_norm = mx;
        std::vector<double> my_norm = my;
        for (double &val : mx_norm)
            val -= originX;
        for (double &val : my_norm)
            val -= originY;

        // Výpočet rychlosti GT pro ANEES
        std::vector<double> vx_gt(N), vy_gt(N);
        double dt = 1.0;
        for (size_t k = 0; k < N - 1; ++k)
        {
            vx_gt[k] = (mx_norm[k + 1] - mx_norm[k]) / dt;
            vy_gt[k] = (my_norm[k + 1] - my_norm[k]) / dt;
        }
        vx_gt[N - 1] = vx_gt[N - 2];
        vy_gt[N - 1] = vy_gt[N - 2];

        // =================================================================
        // KROK B: DEFINICE PARAMETRŮ A GENEROVÁNÍ DAT
        // =================================================================
        double q = 0.1;
        double dt2 = dt * dt;
        double dt3 = dt * dt * dt;

        gtsam::Matrix Q_mat = gtsam::Matrix::Zero(4, 4);
        Q_mat(0, 0) = dt3 / 3.0;
        Q_mat(0, 2) = dt2 / 2.0;
        Q_mat(1, 1) = dt3 / 3.0;
        Q_mat(1, 3) = dt2 / 2.0;
        Q_mat(2, 0) = dt2 / 2.0;
        Q_mat(2, 2) = dt;
        Q_mat(3, 1) = dt2 / 2.0;
        Q_mat(3, 3) = dt;
        Q_mat = Q_mat * q;
        auto procNoise = gtsam::noiseModel::Gaussian::Covariance(Q_mat);

        // 2. Matice měření R

        // auto measNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(5.0, 0.1, 0.1));
        auto measNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(0.5, 0.1));

        auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector4(1.0, 1.0, 0.01, 0.01));
        std::default_random_engine generator(12); // Pevný seed
        std::normal_distribution<double> distMeasV(0.0, 0.1);
        std::normal_distribution<double> distMeasX(0.0, 0.5);

        vector<gtsam::Vector> measurements(N);
        for (size_t k = 0; k < N; ++k)
        {
            double dx = vx_gt[k];
            double dy = vy_gt[k];

            // gtsam::Vector m(3);
            gtsam::Vector m(2);
            //  Pseudo-rychlosti: Přidáváme šum odpovídající R

            // linearni mereni rychlosti
            // m << map.getZ(mx_norm[k], my_norm[k]) + distMeasX(generator), // hB_data[k],
            //    dx + distMeasV(generator),
            //    dy + distMeasV(generator);

            // NElinearni mereni rychlosti
            m << map.getZ(mx_norm[k], my_norm[k]) + distMeasX(generator), // hB_data[k],
                sqrt(dx * dx + dy * dy) + distMeasV(generator);

            measurements[k] = m;
        }

        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initialEstimate;

        std::normal_distribution<double> distP0X(0.0, 1.0);
        std::normal_distribution<double> distP0V(0.0, 0.01);

        gtsam::Vector x0(4);
        x0 << mx_norm[0] + distP0X(generator),
            my_norm[0] + distP0X(generator),
            vx_gt[0] + distP0V(generator),
            vy_gt[0] + distP0V(generator);

        graph.add(gtsam::PriorFactor<gtsam::Vector>(gtsam::Symbol('x', 0), x0, priorNoise));
        initialEstimate.insert(gtsam::Symbol('x', 0), x0);

        for (size_t k = 1; k < N; ++k)
        {
            gtsam::Symbol prevKey('x', k - 1);
            gtsam::Symbol currKey('x', k);

            graph.add(DynamicsFactor(prevKey, currKey, dt, procNoise));
            graph.add(CustomMeasurementFactor(currKey, measurements[k], measNoise, map));

            gtsam::Vector prevDist = initialEstimate.at<gtsam::Vector>(prevKey);
            gtsam::Vector est(4);
            est << prevDist(0) + prevDist(2) * dt,
                prevDist(1) + prevDist(3) * dt,
                prevDist(2),
                prevDist(3);
            initialEstimate.insert(currKey, est);
        }

        // =================================================================
        // KROK C: OPTIMALIZACE
        // =================================================================
        gtsam::LevenbergMarquardtParams params;
        params.maxIterations = 500;
        params.relativeErrorTol = 1e-8;
        params.absoluteErrorTol = 1e-3;
        params.setVerbosity("SILENT");
        params.lambdaInitial = 1e-4;
        params.lambdaFactor = 2;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
        gtsam::Values result = optimizer.optimize();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
        cout << "Počet iterací: " << optimizer.iterations() << endl;
        cout << "Konečná chyba: " << optimizer.error() << endl;

        // =================================================================
        // KROK C.2: VÝPOČET ANEES
        // =================================================================
        cout << "Pocitam kovariance a ANEES..." << endl;

        // Inicializace objektu pro výpočet marginálních kovariancí
        gtsam::Marginals marginals(graph, result);

        double sum_nees = 0.0;

        for (size_t k = 0; k < N; ++k)
        {
            // 1. Získání odhadu a Ground Truth
            gtsam::Vector x_est = result.at<gtsam::Vector>(gtsam::Symbol('x', k));
            gtsam::Vector x_true(4);
            x_true << mx_norm[k], my_norm[k], vx_gt[k], vy_gt[k];

            // 2. Chybový vektor (e)
            gtsam::Vector err = x_true - x_est;

            // 3. Kovarianční matice (P) pro stav k
            gtsam::Matrix P_k = marginals.marginalCovariance(gtsam::Symbol('x', k));

            // 4. Výpočet NEES = err^T * P^-1 * err
            // V Eigen/GTSAM se to dá stabilně řešit pomocí P.ldlt().solve(err) což odpovídá P \ err v MATLABu
            gtsam::Vector P_inv_err = P_k.ldlt().solve(err);
            double nees = err.transpose() * P_inv_err;

            sum_nees += nees;
        }

        double anees = sum_nees / N;
        cout << "=========================================" << endl;
        cout << "VYSLEDEK KONZISTENCE (ANEES v C++)" << endl;
        cout << "Ocekavana hodnota: " << 4.0 << endl;
        cout << "Spocitane ANEES:   " << anees << endl;
        cout << "=========================================" << endl;

        // =================================================================
        // KROK D: DENORMALIZACE A ULOŽENÍ
        // =================================================================
        ofstream out("vysledky_gtsam_lin_mer.csv");
        for (size_t k = 0; k < N; ++k)
        {
            gtsam::Vector local_res = result.at<gtsam::Vector>(gtsam::Symbol('x', k));

            out << std::fixed << std::setprecision(3)
                << k << ","
                << local_res(0) + originX << ","
                << local_res(1) + originY << ","
                << local_res(2) << ","
                << local_res(3) << ","
                << mx[k] << ","
                << my[k] << ","
                << measurements[k][0] << ","
                //<< measurements[k][1] << ","
                << measurements[k][1] << endl;
        }
        cout << "Hotovo. Vysledky ulozeny s puvodnim meritkem." << endl;
    }
    catch (std::exception &e)
    {
        cerr << "CHYBA: " << e.what() << endl;
    }
    return 0;
}