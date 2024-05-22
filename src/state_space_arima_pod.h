//
// Created by alex on 5/22/24.
//

#ifndef ARIMA_STATE_SPACE_ARIMA_POD_H
#define ARIMA_STATE_SPACE_ARIMA_POD_H


#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class ARIMA {
public:
    ARIMA(int p, int d, int q) : p(p), d(d), q(q) {
        // Initialize state-space matrices
        F = vector<vector<double>>(p, vector<double>(p, 0.0));
        for (int i = 1; i < p; ++i) {
            F[i][i - 1] = 1.0;
        }
        H = vector<double>(p, 0.0);
        H[0] = 1.0;
        Q = vector<vector<double>>(p, vector<double>(p, 0.0));
        for (int i = 0; i < p; ++i) {
            Q[i][i] = 1.0;
        }
        R = 1.0;
    }

    void fit(const vector<double>& data) {
        int n = data.size();
        vector<double> Y = data;

        // Difference the data if d > 0
        if (d > 0) {
            Y = difference(Y, d);
        }

        vector<double> x(p, 0.0);
        vector<vector<double>> P(p, vector<double>(p, 0.0));
        for (int i = 0; i < p; ++i) {
            P[i][i] = 1.0;
        }

        for (int t = 0; t < n; ++t) {
            vector<double> x_pred = mat_vec_mult(F, x);
            vector<vector<double>> P_pred = mat_add(mat_mult(F, P, F), Q);

            double y = Y[t];
            double y_pred = vec_dot(H, x_pred);
            double e = y - y_pred;
            double S = vec_dot(H, mat_vec_mult(P_pred, H)) + R;
            vector<double> K = scalar_mult(mat_vec_mult(P_pred, H), 1 / S);

            for (int i = 0; i < p; ++i) {
                x[i] = x_pred[i] + K[i] * e;
            }

            vector<vector<double>> KH = vec_outer(K, H);
            vector<vector<double>> P_update = mat_sub(P_pred, mat_mult(KH, P_pred));

            P = P_update;
            estimated_states.push_back(x);
            estimated_covariances.push_back(P);
        }
    }

    vector<double> forecast(int steps) {
        vector<double> forecasts;

        vector<double> x = estimated_states.back();
        vector<vector<double>> P = estimated_covariances.back();

        for (int i = 0; i < steps; ++i) {
            x = mat_vec_mult(F, x);
            P = mat_add(mat_mult(F, P, F), Q);
            forecasts.push_back(x[0]);
        }

        return forecasts;
    }

private:
    int p, d, q;
    vector<vector<double>> F, Q;
    vector<double> H;
    double R;
    vector<vector<double>> estimated_states;
    vector<vector<vector<double>>> estimated_covariances;

    vector<double> difference(const vector<double>& data, int d) {
        vector<double> diff = data;
        for (int i = 0; i < d; ++i) {
            vector<double> temp(diff.size() - 1);
            for (size_t j = 0; j < diff.size() - 1; ++j) {
                temp[j] = diff[j + 1] - diff[j];
            }
            diff = temp;
        }
        return diff;
    }

    vector<double> mat_vec_mult(const vector<vector<double>>& mat, const vector<double>& vec) {
        vector<double> result(mat.size(), 0.0);
        for (size_t i = 0; i < mat.size(); ++i) {
            for (size_t j = 0; j < mat[i].size(); ++j) {
                result[i] += mat[i][j] * vec[j];
            }
        }
        return result;
    }

    double vec_dot(const vector<double>& vec1, const vector<double>& vec2) {
        double result = 0.0;
        for (size_t i = 0; i < vec1.size(); ++i) {
            result += vec1[i] * vec2[i];
        }
        return result;
    }

    vector<double> scalar_mult(const vector<double>& vec, double scalar) {
        vector<double> result(vec.size(), 0.0);
        for (size_t i = 0; i < vec.size(); ++i) {
            result[i] = vec[i] * scalar;
        }
        return result;
    }

    vector<vector<double>> mat_add(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2) {
        vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0.0));
        for (size_t i = 0; i < mat1.size(); ++i) {
            for (size_t j = 0; j < mat1[i].size(); ++j) {
                result[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
        return result;
    }

    vector<vector<double>> mat_sub(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2) {
        vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0.0));
        for (size_t i = 0; i < mat1.size(); ++i) {
            for (size_t j = 0; j < mat1[i].size(); ++j) {
                result[i][j] = mat1[i][j] - mat2[i][j];
            }
        }
        return result;
    }

    vector<vector<double>> mat_mult(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2, const vector<vector<double>>& mat3) {
        size_t rows = mat1.size();
        size_t cols = mat2[0].size();
        vector<vector<double>> result(rows, vector<double>(cols, 0.0));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                for (size_t k = 0; k < mat1[0].size(); ++k) {
                    result[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }
        return mat_add(result, mat3);
    }

    vector<vector<double>> vec_outer(const vector<double>& vec1, const vector<double>& vec2) {
        vector<vector<double>> result(vec1.size(), vector<double>(vec2.size(), 0.0));
        for (size_t i = 0; i < vec1.size(); ++i) {
            for (size_t j = 0; j < vec2.size(); ++j) {
                result[i][j] = vec1[i] * vec2[j];
            }
        }
        return result;
    }
};

int main() {
    // Example usage
    vector<double> data = { /* your time series data */ };

    int p = 1;  // AR order
    int d = 1;  // Differencing order
    int q = 1;  // MA order

    ARIMA model(p, d, q);
    model.fit(data);

    int forecast_steps = 10;
    vector<double> forecasts = model.forecast(forecast_steps);

    cout << "Forecasts: ";
    for (double f : forecasts) {
        cout << f << " ";
    }
    cout << endl;

    return 0;
}



#endif //ARIMA_STATE_SPACE_ARIMA_POD_H
