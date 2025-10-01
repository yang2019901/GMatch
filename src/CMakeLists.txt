cmake_minimum_required(VERSION 3.12)
project(gmatch_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
add_library(gmatch_cpp SHARED gmatch.cpp)
target_link_libraries(gmatch_cpp pybind11::module Eigen3::Eigen)
set_target_properties(gmatch_cpp PROPERTIES PREFIX "" SUFFIX ".so")