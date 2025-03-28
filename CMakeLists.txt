cmake_minimum_required(VERSION 3.22)
project(OpenMPLab)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_executable(sequential sequential.cpp)
add_executable(parallelMP parallel_mp.cpp)
add_executable(parallelMPI parallel_mpi.cpp)

if (OpenMP_CXX_FOUND)
    target_link_libraries(parallelMP PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(parallelMPI PRIVATE MPI::MPI_CXX)