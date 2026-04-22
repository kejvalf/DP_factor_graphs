cmake_minimum_required(VERSION 3.0)
project(GtsamCustomFactor)

find_package(GTSAM REQUIRED)

# Povolení C++11 nebo novějšího
set(CMAKE_CXX_STANDARD 14)

add_executable(TAN main.cpp)
target_link_libraries(TAN gtsam)