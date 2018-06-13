#ifndef RECOMMEND_CU
#define RECOMMEND_CU

// #include "recommend.h"
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <cmath>
#include <cstring>

double rmse(cublasHandle_t handle, double* user_factors, double* item_factors, int* rows, int* cols, double* ratings, int num_things, int factors);

#endif
