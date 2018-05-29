cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(Recommendation)
find_package(CUDA)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
CUDA_ADD_EXECUTABLE(cRecommendation main.cu recommend.cu recommend.h)

