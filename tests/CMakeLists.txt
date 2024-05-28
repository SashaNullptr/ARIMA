include(FetchContent)

find_package( Eigen3 3.3.4 REQUIRED CONFIG )
include_directories(${EIGEN3_INCLUDE_DIR})

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.14.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(arima_eigen_test arima_eigen_test.cpp)

target_link_libraries(arima_eigen_test
        PRIVATE
        GTest::GTest)

add_test(arima_eigen_gtest arima_eigen_test)