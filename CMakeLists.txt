cmake_minimum_required(VERSION 3.22)
project(OpenMPLab)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(OpenMPLab main.cpp
        sequential.cpp)
