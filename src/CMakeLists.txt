#target_include_directories(arima PUBLIC include)

find_package( Eigen3 3.3.4 REQUIRED CONFIG )
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(arima_state_space_eigen STATIC state_space_arima_eigen.cpp state_space_arima_eigen.h)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

target_include_directories(arima_state_space_eigen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})