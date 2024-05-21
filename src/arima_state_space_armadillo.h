#ifndef ARIMA_ARIMA_STATE_SPACE_ARMADILLO_H
#define ARIMA_ARIMA_STATE_SPACE_ARMADILLO_H


#include <iostream>
#include <std::vector>
#include <armadillo>

using namespace arma;

class ARIMA {
public:
    ARIMA(int p, int d, int q) : p(p), d(d), q(q) {
        // Initialize state-space matrices
        F = mat(p, p, fill::zeros);
        for (int i = 1; i < p; ++i) {
            F(i, i - 1) = 1.0;
        }
        H = vec(p, fill::zeros);
        H(0) = 1.0;
        Q = mat(p, p, fill::eye);
        R = mat(1, 1, fill::eye);
    }

    void fit(const std::vector<double>& data) {
        int n = data.size();
        vec Y = vec(data);

        // Difference the data if d > 0
        if (d > 0) {
            Y = difference(Y, d);
        }

        vec x = vec(p, fill::zeros);
        mat P = mat(p, p, fill::eye);

        for (int t = 0; t < n; ++t) {
            vec x_pred = F * x;
            mat P_pred = F * P * F.t() + Q;

            vec y(1);
            y(0) = Y(t);
            vec y_pred = H.t() * x_pred;
            vec e = y - y_pred;
            mat S = H.t() * P_pred * H + R;
            mat K = P_pred * H * inv(S);

            x = x_pred + K * e;
            P = P_pred - K * H.t() * P_pred;

            estimated_states.push_back(x);
            estimated_covariances.push_back(P);
        }
    }

    std::vector<double> forecast(int steps) {
        std::vector<double> forecasts;

        vec x = estimated_states.back();
        mat P = estimated_covariances.back();

        for (int i = 0; i < steps; ++i) {
            x = F * x;
            P = F * P * F.t() + Q;
            forecasts.push_back(x(0));
        }

        return forecasts;
    }

private:
    int p, d, q;
    mat F, Q, R;
    vec H;
    std::vector<vec> estimated_states;
    std::vector<mat> estimated_covariances;

    vec difference(const vec& data, int d) {
        vec diff = data;
        for (int i = 0; i < d; ++i) {
            diff = diff.tail(diff.n_elem - 1) - diff.head(diff.n_elem - 1);
        }
        return diff;
    }
};


#endif //ARIMA_ARIMA_STATE_SPACE_ARMADILLO_H
