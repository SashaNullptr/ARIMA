#ifndef ARIMA_STATE_SPACE_EIGEN_H
#define ARIMA_STATE_SPACE_EIGEN_H


#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class ARIMA {
public:
    ARIMA(int p, int d, int q) : p(p), d(d), q(q) {
        // Initialize state-space matrices
        F = MatrixXd::Zero(p, p);
        for (int i = 1; i < p; ++i) {
            F(i, i - 1) = 1.0;
        }
        H = VectorXd::Zero(p);
        H(0) = 1.0;
        Q = MatrixXd::Identity(p, p);
        R = MatrixXd::Identity(1, 1);
    }

    void fit(const std::vector<vector>& data) {
        int n = data.size();
        VectorXd Y = VectorXd::Map(data.data(), n);

        // Difference the data if d > 0
        if (d > 0) {
            Y = difference(Y, d);
        }

        VectorXd x = VectorXd::Zero(p);
        MatrixXd P = MatrixXd::Identity(p, p);

        for (int t = 0; t < n; ++t) {
            VectorXd x_pred = F * x;
            MatrixXd P_pred = F * P * F.transpose() + Q;

            VectorXd y = Y.segment(t, 1);
            VectorXd y_pred = H.transpose() * x_pred;
            VectorXd e = y - y_pred;
            MatrixXd S = H.transpose() * P_pred * H + R;
            MatrixXd K = P_pred * H * S.inverse();

            x = x_pred + K * e;
            P = P_pred - K * H.transpose() * P_pred;

            estimated_states.push_back(x);
            estimated_covariances.push_back(P);
        }
    }

    std::vector<vector> forecast(int steps) {
        std::vector<vector> forecasts;

        VectorXd x = estimated_states.back();
        MatrixXd P = estimated_covariances.back();

        for (int i = 0; i < steps; ++i) {
            x = F * x;
            P = F * P * F.transpose() + Q;
            forecasts.push_back(x(0));
        }

        return forecasts;
    }

private:
    int p, d, q;
    MatrixXd F, Q, R;
    VectorXd H;
    vector<VectorXd> estimated_states;
    vector<MatrixXd> estimated_covariances;

    VectorXd difference(const VectorXd& data, int d) {
        VectorXd diff = data;
        for (int i = 0; i < d; ++i) {
            diff = diff.tail(diff.size() - 1) - diff.head(diff.size() - 1);
        }
        return diff;
    }
};

//int main() {
//    std::vector<vector> data = { /* your time series data */ };
//
//    int p = 1;
//    int d = 1;
//    int q = 1;
//
//    ARIMA model(p, d, q);
//    model.fit(data);
//
//    int forecast_steps = 10;
//    std::vector<vector> forecasts = model.forecast(forecast_steps);
//
//    cout << "Forecasts: ";
//    for (double f : forecasts) {
//        cout << f << " ";
//    }
//    cout << endl;
//
//    return 0;
//}

#endif //ARIMA_STATE_SPACE_EIGEN_H
