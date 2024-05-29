#include <gtest/gtest.h>
#include "../src/state_space_arima_eigen.h"

extern "C"

TEST(ARIMAModelTest, FitAndForecastNoMissingData) {
    // Sample data without missing values
    std::vector<double> data = { 1.2, 1.8, 2.5, 3.1, 3.9, 4.2, 5.0, 5.3, 5.8, 6.1 };

    // Create ARIMA model with p=1, d=0, q=1
    ARIMAModel arima(1, 0, 1);

    // Fit the model to data
    arima.fit(data);

    // Forecast next 5 values
    std::vector<double> forecasts = arima.forecast(5);

    // Print forecasts
    std::cout << "Forecasts (No Missing Data): ";
    for (const auto& value : forecasts) {
        if (std::isnan(value)) {
            std::cout << "NaN ";
        } else {
            std::cout << value << " ";
        }
    }
    std::cout << std::endl;

    // Add assertions for expected behavior
    ASSERT_EQ(forecasts.size(), 5);
}

TEST(ARIMAModelTest, FitAndForecastNoMissingDataDEqualOne) {
    // Sample data without missing values
    std::vector<double> data = { 1.2, 1.8, 2.5, 3.1, 3.9, 4.2, 5.0, 5.3, 5.8, 6.1 };

    // Create ARIMA model with p=1, d=0, q=1
    ARIMAModel arima(1, 1, 1);

    // Fit the model to data
    arima.fit(data);

    // Forecast next 5 values
    std::vector<double> forecasts = arima.forecast(5);

    // Print forecasts
    std::cout << "Forecasts (No Missing Data, d = 1): ";
    for (const auto& value : forecasts) {
        if (std::isnan(value)) {
            std::cout << "NaN ";
        } else {
            std::cout << value << " ";
        }
    }
    std::cout << std::endl;

    // Add assertions for expected behavior
    ASSERT_EQ(forecasts.size(), 6);
}

TEST(ARIMAModelTest, FitAndForecastConsecutiveMissingData) {
    // Sample data with consecutive missing values
    std::vector<double> data = { 1.2, 1.8, std::nan(""), std::nan(""), 3.9, 4.2, 5.0, 5.3, 5.8, 6.1 };

    // Create ARIMA model with p=1, d=0, q=1
    ARIMAModel arima(3, 0, 1);

    // Fit the model to data
    arima.fit(data);

    // Forecast next 5 values
    std::vector<double> forecasts = arima.forecast(5);

    // Print forecasts
    std::cout << "Forecasts (Consecutive Missing Data): ";
    for (const auto& value : forecasts) {
        if (std::isnan(value)) {
            std::cout << "NaN ";
        } else {
            std::cout << value << " ";
        }
    }
    std::cout << std::endl;

    // Add assertions for expected behavior
    ASSERT_EQ(forecasts.size(), 5);
}

TEST(ARIMAModelTest, FitAndForecastStartMissingData) {
    // Sample data with missing values at the start
    std::vector<double> data = { std::nan(""), std::nan(""), 2.5, 3.1, 3.9, 4.2, 5.0, 5.3, 5.8, 6.1 };

    // Create ARIMA model with p=1, d=0, q=1
    ARIMAModel arima(3, 0, 1);

    // Fit the model to data
    arima.fit(data);

    // Forecast next 5 values
    std::vector<double> forecasts = arima.forecast(5);

    // Print forecasts
    std::cout << "Forecasts (Start Missing Data): ";
    for (const auto& value : forecasts) {
        if (std::isnan(value)) {
            std::cout << "NaN ";
        } else {
            std::cout << value << " ";
        }
    }
    std::cout << std::endl;

    // Add assertions for expected behavior
    ASSERT_EQ(forecasts.size(), 5);
}

TEST(ARIMAModelTest, FitAndForecastEndMissingData) {
    // Sample data with missing values at the end
    std::vector<double> data = { 1.2, 1.8, 2.5, 3.1, 3.9, 4.2, 5.0, std::nan(""), std::nan(""), std::nan("") };

    // Create ARIMA model with p=1, d=0, q=1
    ARIMAModel arima(3, 0, 1);

    // Fit the model to data
    arima.fit(data);

    // Forecast next 5 values
    std::vector<double> forecasts = arima.forecast(5);

    // Print forecasts
    std::cout << "Forecasts (End Missing Data): ";
    for (const auto& value : forecasts) {
        if (std::isnan(value)) {
            std::cout << "NaN ";
        } else {
            std::cout << value << " ";
        }
    }
    std::cout << std::endl;

    // Add assertions for expected behavior
    ASSERT_EQ(forecasts.size(), 5);
}

TEST(ARIMAModelTest, FitAndForecastRandomMissingData) {
    // Sample data with randomly scattered missing values
    std::vector<double> data = { 1.2, std::nan(""), 2.5, 3.1, std::nan(""), 4.2, std::nan(""), 5.3, 5.8, 6.1 };

    // Create ARIMA model with p=1, d=0, q=1
    ARIMAModel arima(3, 1, 1);

    // Fit the model to data
    arima.fit(data);

    // Forecast next 5 values
    std::vector<double> forecasts = arima.forecast(5);

    // Print forecasts
    std::cout << "Forecasts (Random Missing Data): ";
    for (const auto& value : forecasts) {
        if (std::isnan(value)) {
            std::cout << "NaN ";
        } else {
            std::cout << value << " ";
        }
    }
    std::cout << std::endl;

    // Add assertions for expected behavior
    ASSERT_EQ(forecasts.size(), 5);
}