#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>

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

        double v_norm_sq = vx * vx + vy * vy;
        double v_norm = sqrt(v_norm_sq);
        if (v_norm < 1e-4)
            v_norm = 1e-4;
        double v_norm3 = v_norm_sq * v_norm;

        double h_baro = map_.getZ(px, py);
        double h_vx = (vx * vx - vy * vy) / v_norm;
        double h_vy = (2 * vx * vy) / v_norm;

        gtsam::Vector h_x(3);
        h_x << h_baro, h_vx, h_vy;

        if (H)
        {
            gtsam::Matrix J = gtsam::Matrix::Zero(3, 4);

            double dZdx, dZdy;
            map_.getGradients(px, py, dZdx, dZdy);
            J(0, 0) = dZdx;
            J(0, 1) = dZdy;
            J(1, 2) = (vx * vx * vx + 3 * vx * vy * vy) / v_norm3;
            J(1, 3) = -(3 * vx * vx * vy + vy * vy * vy) / v_norm3;
            J(2, 2) = (2 * vy * vy * vy) / v_norm3;
            J(2, 3) = (2 * vx * vx * vx) / v_norm3;

            *H = J;
        }
        return h_x - measured_;
    }
};

int main()
{
    try
    {
        // -----------------------------------------------------------------
        // 1. NAČTENÍ DAT (BEZ NORMALIZACE)
        // -----------------------------------------------------------------
        auto raw_x = flatten(readCSV("../mapX.csv"));
        auto raw_y = flatten(readCSV("../mapY.csv"));
        auto raw_z = readCSV("../mapZ.csv");
        auto hB_data = flatten(readCSV("../hB.csv"));

        // Ground Truth Pozice (načteno přímo, bez posunu)
        std::vector<double> mx = flatten(readCSV("../GNSS_X.csv"));
        std::vector<double> my = flatten(readCSV("../GNSS_Y.csv"));

        size_t N = min(mx.size(), hB_data.size());
        if (N == 0)
            return 0;

        // Inicializace mapy s původními souřadnicemi
        MapInterpolant map(raw_x, raw_y, raw_z);

        // Ground Truth Rychlosti
        std::vector<double> vx_gt(N), vy_gt(N);
        double dt = 1.0;
        for (size_t k = 0; k < N - 1; ++k)
        {
            vx_gt[k] = (mx[k + 1] - mx[k]) / dt;
            vy_gt[k] = (my[k + 1] - my[k]) / dt;
        }
        vx_gt[N - 1] = vx_gt[N - 2];
        vy_gt[N - 1] = vy_gt[N - 2];

        // -----------------------------------------------------------------
        // 2. DEFINICE SYSTÉMU
        // -----------------------------------------------------------------
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

        auto measNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(5.0, 0.5, 0.5));
        auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector4(1.0, 1.0, 1.0, 1.0));

        // -----------------------------------------------------------------
        // 3. NASTAVENÍ MONTE CARLO
        // -----------------------------------------------------------------
        const int N_MC = 50; // Počet iterací
        const double RMSE_THRESHOLD = 100.0;

        // Struktura pro uložení výsledků
        struct Result
        {
            int id;
            double rmse_total;
            double time_ms;
            size_t iterations; // Počet iterací optimizeru
            bool success;
        };
        vector<Result> results;
        int success_count = 0;

        cout << "Spoustim Monte Carlo simulaci (C++ GTSAM, bez normalizace, " << N_MC << " behu)..." << endl;
        cout << "--------------------------------------------------------------------" << endl;
        cout << "Iter | Total RMSE [?] | Cas [ms] | Iters | Stav" << endl;
        cout << "--------------------------------------------------------------------" << endl;

        // -----------------------------------------------------------------
        // 4. HLAVNÍ SMYČKA MC
        // -----------------------------------------------------------------
        for (int i_mc = 0; i_mc < N_MC; ++i_mc)
        {
            std::mt19937 generator(i_mc);
            std::normal_distribution<double> n_meas_vel(0.0, 0.5);
            std::normal_distribution<double> n_prior(0.0, 1.0);

            // A) Generování měření
            vector<gtsam::Vector> measurements(N);
            for (size_t k = 0; k < N; ++k)
            {
                double v_curr = sqrt(vx_gt[k] * vx_gt[k] + vy_gt[k] * vy_gt[k]);
                if (v_curr < 1e-4)
                    v_curr = 1e-4;

                double h_vx = (vx_gt[k] * vx_gt[k] - vy_gt[k] * vy_gt[k]) / v_curr;
                double h_vy = (2 * vx_gt[k] * vy_gt[k]) / v_curr;

                gtsam::Vector m(3);
                m << hB_data[k],
                    h_vx + n_meas_vel(generator),
                    h_vy + n_meas_vel(generator);
                measurements[k] = m;
            }

            // B) Graf
            gtsam::NonlinearFactorGraph graph;
            gtsam::Values initialEstimate;

            gtsam::Vector x0(4);
            x0 << mx[0] + n_prior(generator),
                my[0] + n_prior(generator),
                vx_gt[0] + n_prior(generator),
                vy_gt[0] + n_prior(generator);

            graph.add(gtsam::PriorFactor<gtsam::Vector>(gtsam::Symbol('x', 0), x0, priorNoise));
            initialEstimate.insert(gtsam::Symbol('x', 0), x0);

            gtsam::Vector current_state = x0;
            for (size_t k = 1; k < N; ++k)
            {
                gtsam::Symbol prevKey('x', k - 1);
                gtsam::Symbol currKey('x', k);

                graph.add(DynamicsFactor(prevKey, currKey, dt, procNoise));
                graph.add(CustomMeasurementFactor(currKey, measurements[k], measNoise, map));

                // Predikce pro initial guess
                current_state(0) += current_state(2) * dt;
                current_state(1) += current_state(3) * dt;
                initialEstimate.insert(currKey, current_state);
            }

            // C) Optimalizace
            gtsam::LevenbergMarquardtParams params;
            params.setVerbosity("SILENT");
            params.relativeErrorTol = 1e-5;
            params.maxIterations = 300;

            auto start_time = std::chrono::high_resolution_clock::now();
            gtsam::LevenbergMarquardtOptimizer optimizer(graph, initialEstimate, params);
            gtsam::Values result = optimizer.optimize();
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            // Získání počtu iterací
            size_t num_iters = optimizer.iterations();

            // D) Vyhodnocení
            double sum_sq_total = 0.0;

            for (size_t k = 0; k < N; ++k)
            {
                gtsam::Vector est = result.at<gtsam::Vector>(gtsam::Symbol('x', k));

                double dx = est(0) - mx[k];
                double dy = est(1) - my[k];
                double dvx = est(2) - vx_gt[k];
                double dvy = est(3) - vy_gt[k];

                sum_sq_total += (dx * dx + dy * dy + dvx * dvx + dvy * dvy);
            }
            double rmse_total = sqrt(sum_sq_total / N);

            bool success = (rmse_total < RMSE_THRESHOLD);
            if (success)
                success_count++;

            results.push_back({i_mc, rmse_total, duration, num_iters, success});

            cout << setw(4) << i_mc << " | "
                 << fixed << setprecision(2) << setw(14) << rmse_total << " | "
                 << setw(8) << duration << " | "
                 << setw(5) << num_iters << " | "
                 << (success ? "OK" : "FAIL") << endl;
        }

        // -----------------------------------------------------------------
        // 5. STATISTIKY
        // -----------------------------------------------------------------
        double total_rmse_avg = 0.0;
        double total_time_avg = 0.0;
        size_t total_iterations_sum = 0;
        int valid_runs = 0;

        for (const auto &r : results)
        {
            if (r.success)
            {
                total_rmse_avg += r.rmse_total;
                total_time_avg += r.time_ms;
                total_iterations_sum += r.iterations;
                valid_runs++;
            }
        }

        cout << "--------------------------------------------------------" << endl;
        cout << "VYSLEDKY C++ GTSAM BATCH (TOTAL RMSE):" << endl;
        cout << "Pocet behu: " << N_MC << endl;
        cout << "Uspesnost:  " << success_count << " (" << (double)success_count / N_MC * 100.0 << "%)" << endl;
        if (valid_runs > 0)
        {
            double avg_time = total_time_avg / valid_runs;
            double avg_time_per_iter = (total_iterations_sum > 0) ? (total_time_avg / total_iterations_sum) : 0.0;

            cout << "Prum. RMSE Total: " << total_rmse_avg / valid_runs << endl;
            cout << "Prum. Cas (beh):  " << avg_time << " ms" << endl;
            cout << "Prum. Cas/Iter:   " << avg_time_per_iter << " ms" << endl;
        }
        cout << "--------------------------------------------------------" << endl;
    }
    catch (std::exception &e)
    {
        cerr << "CRITICAL ERROR: " << e.what() << endl;
    }
    return 0;
}