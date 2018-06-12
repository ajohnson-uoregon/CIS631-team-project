#ifndef calculate_loss_CU
#define calculate_loss_CU

#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <cmath>
#include <vector>
#include <cstring>

double calculate_loss(int* indptr, int* indices, double* data, double* X, double* Y, double reg, 
    int users, int items, int factors, int nnz);

#endif