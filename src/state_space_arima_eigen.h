#ifndef ARIMA_STATE_SPACE_EIGEN_H
#define ARIMA_STATE_SPACE_EIGEN_H

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <functional>

// Define ARIMA model class
class ARIMAModel {
public:
    ARIMAModel(int p, int d, int q) : p(p), d(d), q(q) {}

    // Fit the model to data using EM algorithm
    void fit(std::vector<double> data, int maxIter = 100, double tol = 1e-6);

    // Forecast the next values
    std::vector<double> forecast(int steps);

private:
    int p, d, q;
    Eigen::VectorXd arParams; // AR coefficients
    Eigen::VectorXd maParams; // MA coefficients
    double sigma2; // Variance of residuals

    // Log-likelihood function
    double logLikelihood(const Eigen::VectorXd& params);

    // Interpolate missing values in data
    void interpolateMissingValues(std::vector<double>& data);

    // Apply differencing
    std::vector<double> difference(const std::vector<double>& data, int d);
    std::vector<double> undifference(const std::vector<double>& differencedData, const std::vector<double>& originalData, int d);

    std::vector<double> data; // Store the data
};

// Define the log-likelihood function
double ARIMAModel::logLikelihood(const Eigen::VectorXd& params) {
    int n = data.size();
    int maxPQ = std::max(p, q);
    Eigen::VectorXd residuals(n);
    residuals.setZero();

    // Extract parameters
    arParams = params.segment(0, p);
    maParams = params.segment(p, q);
    sigma2 = params(p + q);

    // Compute residuals
    for (int t = maxPQ; t < n; ++t) {
        double predicted = 0.0;
        for (int i = 0; i < p; ++i) {
            predicted += arParams(i) * data[t - i - 1];
        }
        for (int j = 0; j < q; ++j) {
            predicted += maParams(j) * residuals[t - j - 1];
        }
        residuals[t] = data[t] - predicted;
    }

    // Compute log-likelihood
    double logL = -0.5 * n * std::log(2 * M_PI * sigma2);
    logL -= (residuals.array().square().sum()) / (2 * sigma2);
    return logL;
}

// Interpolate missing values in data
void ARIMAModel::interpolateMissingValues(std::vector<double>& data) {
    int n = data.size();
    int start = -1;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(data[i])) {
            if (start == -1) {
                start = i;
            }
        } else if (start != -1) {
            int end = i;
            double step = (data[end] - data[start - 1]) / (end - start + 1);
            for (int j = start; j < end; ++j) {
                data[j] = data[start - 1] + step * (j - start + 1);
            }
            start = -1;
        }
    }
}

// Fit the ARIMA model to data using EM algorithm
void ARIMAModel::fit(std::vector<double> data, int maxIter, double tol) {
    this->data = data;
    interpolateMissingValues(this->data);

    int n = data.size();
    int maxPQ = std::max(p, q);

    // Initial parameter guess
    Eigen::VectorXd params(p + q + 1);
    params.setZero();
    params(p + q) = 1.0; // Initial guess for sigma2

    Eigen::VectorXd oldParams = params;
    double logL = logLikelihood(params);

    for (int iter = 0; iter < maxIter; ++iter) {
        // E-step: Compute residuals and their variances
        Eigen::VectorXd residuals(n);
        residuals.setZero();
        for (int t = maxPQ; t < n; ++t) {
            if (std::isnan(data[t])) {
                continue;
            }
            double predicted = 0.0;
            for (int i = 0; i < p; ++i) {
                if (t - i - 1 >= 0 && !std::isnan(data[t - i - 1])) {
                    predicted += arParams(i) * data[t - i - 1];
                }
            }
            for (int j = 0; j < q; ++j) {
                if (t - j - 1 >= 0 && !std::isnan(residuals[t - j - 1])) {
                    predicted += maParams(j) * residuals[t - j - 1];
                }
            }
            residuals[t] = data[t] - predicted;
        }

        // M-step: Update AR and MA parameters and sigma2
        Eigen::MatrixXd X(n - maxPQ, p + q);
        for (int t = maxPQ; t < n; ++t) {
            if (std::isnan(data[t])) {
                continue;
            }
            for (int i = 0; i < p; ++i) {
                if (t - i - 1 >= 0) {
                    X(t - maxPQ, i) = data[t - i - 1];
                }
            }
            for (int j = 0; j < q; ++j) {
                if (t - j - 1 >= 0) {
                    X(t - maxPQ, p + j) = residuals[t - j - 1];
                }
            }
        }

        Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(data.data() + maxPQ, n - maxPQ);
        Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
        params.segment(0, p + q) = beta;

        double sigma2New = (residuals.segment(maxPQ, n - maxPQ).array().square().sum()) / (n - maxPQ);
        params(p + q) = sigma2New;

        // Check for convergence
        double newLogL = logLikelihood(params);
        if (std::abs(newLogL - logL) < tol) {
            break;
        }

        oldParams = params;
        logL = newLogL;
    }

    // Update model parameters with final values
    arParams = params.segment(0, p);
    maParams = params.segment(p, q);
    sigma2 = params(p + q);

    std::cout << "Final parameters: " << params.transpose() << std::endl;
}

// Forecast future values, supporting missing data
std::vector<double> ARIMAModel::forecast(int steps) {
    std::vector<double> forecasts(steps, std::nan(""));

    int n = data.size();
    std::vector<double> dataCopy = data; // Create a copy of the data

    interpolateMissingValues(dataCopy); // Ensure the copy has no missing values
    // Apply differencing to data
    std::vector<double> differencedData = difference(dataCopy, d);

    for (int t = 0; t < steps; ++t) {
        double predicted = 0.0;
        int availableDataCount = 0;
        for (int i = 0; i < p; ++i) {
            if (n - 1 - i >= 0) {
                predicted += arParams(i) * differencedData[n - 1 - i];
                availableDataCount++;
            }
        }
        for (int j = 0; j < q && t - j - 1 >= 0; ++j) {
            if (!std::isnan(forecasts[t - j - 1])) {
                predicted += maParams(j) * forecasts[t - j - 1];
                availableDataCount++;
            }
        }
        if (availableDataCount > 0) {
            forecasts[t] = predicted / availableDataCount;
        } else {
            forecasts[t] = differencedData.back(); // Fallback to last observed value
        }
        differencedData.push_back(forecasts[t]); // Append forecast to the copy of data
    }

    // Undo differencing to get the final forecasted values
    forecasts = undifference(forecasts, data, d);

    return forecasts;
}

// Apply differencing to data
std::vector<double> ARIMAModel::difference(const std::vector<double>& data, int d) {
    std::vector<double> differencedData = data;
    for (int i = 0; i < d; ++i) {
        for (size_t j = differencedData.size() - 1; j > 0; --j) {
            differencedData[j] = differencedData[j] - differencedData[j - 1];
        }
        differencedData.erase(differencedData.begin());
    }
    return differencedData;
}

// Undo differencing to get the forecasted values
std::vector<double> ARIMAModel::undifference(const std::vector<double>& series, const std::vector<double>& original, int d) {
    std::vector<double> rev = series;
    for (int i = 0; i < d; ++i) {
    std::vector<double> temp(rev.size() + 1);
        temp[0] = original[original.size() - rev.size() - 1];
        for (size_t j = 1; j < temp.size(); ++j) {
            temp[j] = rev[j - 1] + temp[j - 1];
        }
        rev = temp;
    }
    return rev;
}


#endif //ARIMA_STATE_SPACE_EIGEN_H
