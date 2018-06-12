#ifndef LEASTSQR_CU
#define LEASTSQR_CU
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusolverDn.h>
#include <curand.h>
void GPU_fill_rand(double *A, const int nr_rows_A, const int nr_cols_A);
void least_squares (int* indptr, int* indic, double* data, int users, 
                        int items, int factors,
                        double* x, double* y, 
                        double reg, int istart, int iend);

#endif