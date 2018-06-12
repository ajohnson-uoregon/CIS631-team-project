#ifndef calculate_loss_CU
#define calculate_loss_CU

#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstring>

double calculate_loss(int* indptr, int* indices, double* data, double** X, double** Y, double reg, 
    int users, int items, int factors, int nnz) {
  
      int loss = 0;
      int total_confidence = 0;
      int item_norm = 0;
      int user_norm = 0;
      cublasStatus_t err;
      cublasHandle_t handle;
  
      double** YtY;
      const double alpha = 1;
      const double beta = 0;
  
      // do transpose
      err = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, items, factors,
                &alpha, Y[0], items, &beta, Y[0], items, YtY[0], factors);
      
      for (int u = 0; u < users; ++u) {
          double temp = 1.0;
  
          double* r;
          double* Xu = X[u];
          
  
          err = cublasDgemv(handle, CUBLAS_OP_N, items, factors,
                    &alpha, Y[0], items, Xu, 1, &beta, r, 1);
          
          int rowStart = indptr[u];
          int rowEnd = indptr[u+1];
  
          int cols[rowEnd-rowStart];
          memcpy(cols, &indices[rowStart], rowEnd-rowStart);
          //int* cols = Cui[1][rowStart:rowEnd];
          
          double vals[rowEnd-rowStart];
          memcpy(vals, &data[rowStart], rowEnd-rowStart);
          //double* vals = Cui[2][rowStart:rowEnd];
          for (int index = 0; index < rowEnd-rowStart; ++index) {
              int i = cols[index];
              double confidence = vals[index];
  
              double* Yi = Y[i];
  
              double d;
              err = cublasDdot(handle, factors, Yi, 1, Xu, 1, &d);
              temp = (confidence - 1)*d - (2*confidence);
              
              err = cublasDaxpy(handle, factors, &temp, Yi, 1, r, 1);
              total_confidence += confidence;
              loss += confidence;
          }
  
          double other_temp;
  
          err = cublasDdot(handle, factors, r, 1, Xu, 1, &other_temp);
          loss += other_temp;
          
          err = cublasDdot(handle, factors, Xu, 1, Xu, 1, &other_temp);
          user_norm += other_temp;
      }
  
      for (int i = 0; i < items; ++i) {
          double* Yi = Y[i];
  
          double other_temp;
   
          err = cublasDdot(handle, factors, Yi, 1, Yi, 1, &other_temp);
          item_norm += other_temp;
      }
  
      loss += reg * (item_norm + user_norm);
  
      return loss / (total_confidence + users * items - nnz);
  }

#endif
