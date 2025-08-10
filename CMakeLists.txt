cmake_minimum_required(VERSION 3.10)
project(conv_module LANGUAGES CXX)

find_package(pybind11 REQUIRED)
find_package(OpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

pybind11_add_module(conv_module convolution.cpp)
