#include "arima.h"

#include <iostream>

ARIMA::ARIMA(const std::vector<uint8_t>& moving_average_coefs
             ,const uint8_t& difference_term
             ,const std::vector<uint8_t>& autoregression_coef
             ,const uint8_t& constant_offset) :
             moving_average_coefs(moving_average_coefs)
             ,difference_term(difference_term)
             ,autoregression_coef(autoregression_coef)
             ,constant_offset(constant_offset)
             {}

std::vector<uint8_t> ARIMA::difference(const std::vector<uint8_t>& data, uint8_t lag) {
    std::vector<uint8_t> diff;
    for (size_t i = lag; i < data.size(); ++i) {
        diff.push_back(data[i] - data[i - lag]);
    }
    return diff;
}

std::vector<uint8_t> ARIMA::predict_nth(const std::vector<uint8_t>& data, const uint8_t& num_steps) {
    std::vector<uint8_t> diff_data = difference(data, difference_term);
    std::vector<uint8_t> forecast_data = diff_data;

    size_t p = autoregression_coef.size();
    size_t q = moving_average_coefs.size();

    for (uint8_t step = 0; step < num_steps; ++step) {
        uint8_t ar_part = 0;
        uint8_t ma_part = 0;

        // AR part
        for (int i = 0; i < p; ++i) {
            if (forecast_data.size() > i) {
                ar_part += autoregression_coef[i] * forecast_data[forecast_data.size() - 1 - i];
            }
        }

        // MA part
        for (int i = 0; i < q; ++i) {
            if (forecast_data.size() > i) {
                ma_part += moving_average_coefs[i] * (forecast_data[forecast_data.size() - 1 - i] - constant_offset);
            }
        }

        uint8_t next_forecast = constant_offset + ar_part + ma_part;
        forecast_data.push_back(next_forecast);
    }

    return forecast_data;
}