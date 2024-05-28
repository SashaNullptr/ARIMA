cmake_minimum_required(VERSION 3.28)
project(arima CXX C)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)