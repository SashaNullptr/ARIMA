add_library(arima arima.cpp)
target_include_directories(arima PUBLIC include)

find_package( Eigen3 3.3.4 REQUIRED CONFIG )

add_library(arima_state_space_eigen state_space_arima_eigen.cpp)
target_include_directories(arima_state_space_eigen PUBLIC include)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})