cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(Recommendation)
find_package(CUDA)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-O2;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
CUDA_ADD_EXECUTABLE(Recommendation main.cpp recommend.cpp recommend.cu)

