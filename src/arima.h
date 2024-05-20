#ifndef ARIMA_H
#define ARIMA_H

#include <cstdint>
#include <vector>

class ARIMA{
public:
    ARIMA(
            const std::vector<uint8_t>& moving_average_coefs
            ,const uint8_t& difference_term
            ,const std::vector<uint8_t>& autoregression_coef
            ,const uint8_t& constant_offset
    );
    std::vector<uint8_t> predict_nth(const std::vector<uint8_t>& data, const uint8_t& num_steps);
private:
    static std::vector<uint8_t> difference(const std::vector<uint8_t>& data, uint8_t lag);
    std::vector<uint8_t> moving_average_coefs;
    uint8_t difference_term;
    std::vector<uint8_t> autoregression_coef;
    uint8_t constant_offset;
};
#endif //ARIMA_H
