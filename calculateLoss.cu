#include "calculateLoss.h"

double calculate_loss(cublasHandle_t handle, int* indptr, int* indices, double* data, double* X, double* Y, double reg,
    int users, int items, int factors, int nnz) {

      int loss = 0;
      int total_confidence = 0;
      int item_norm = 0;
      int user_norm = 0;
      cublasStatus_t stat;
      cudaError_t err;

      // malloc this
      double* YtY;
      err = cudaMallocManaged(&YtY, factors*factors*sizeof(double));
      if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        cudaFree(YtY);
        return -1;
      }
      const double alpha = 1;
      const double beta = 0;

      // do transpose
      stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, factors, factors, items,
                &alpha, Y, items, Y, items, &beta, YtY, factors);
      if (stat != CUBLAS_STATUS_SUCCESS) {
          printf ("dgemm failed\n");
          cudaFree(YtY);
          return -1;
      }

      for (int u = 0; u < users; ++u) {
        if (u % 10000 == 0) {
          printf("loss iter %d\n", u);
        }
          cudaDeviceSynchronize();
          double temp = 1.0;

          double* r;
          err = cudaMallocManaged(&r, items*sizeof(double));
          if (err != cudaSuccess) {
            printf("%s\n", cudaGetErrorString(err));
            cudaFree(YtY);
            cudaFree(r);
            return -1;
          }
          double* Xu = &X[u*factors];

          stat = cublasDgemv(handle, CUBLAS_OP_N, items, factors,
                    &alpha, Y, items, Xu, 1, &beta, r, 1);
          if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("dgemv failed\n");
              cudaFree(YtY);
              cudaFree(r);
              return -1;
          }
          cudaDeviceSynchronize();

          int rowStart = indptr[u];
          int rowEnd = indptr[u+1];

          int cols[rowEnd-rowStart];
          memcpy(cols, &indices[rowStart], (rowEnd-rowStart)*sizeof(int));
          //int* cols = Cui[1][rowStart:rowEnd];

          double vals[rowEnd-rowStart];
          memcpy(vals, &data[rowStart], (rowEnd-rowStart)*sizeof(double));
          //double* vals = Cui[2][rowStart:rowEnd];
          for (int index = 0; index < rowEnd-rowStart; ++index) {
              int i = cols[index];
              double confidence = vals[index];

              double* Yi = &Y[i*factors];
              cudaDeviceSynchronize();
              double d;
              stat = cublasDdot(handle, factors, Yi, 1, Xu, 1, &d);
              if (stat != CUBLAS_STATUS_SUCCESS) {
                  printf ("ddot 1 failed\n");
                  cudaFree(YtY);
                  cudaFree(r);
                  return -1;
              }
              temp = (confidence - 1)*d - (2*confidence);
              cudaDeviceSynchronize();
              stat = cublasDaxpy(handle, factors, &temp, Yi, 1, r, 1);
              if (stat != CUBLAS_STATUS_SUCCESS) {
                  printf ("daxpy failed\n");
                  cudaFree(YtY);
                  cudaFree(r);
                  return -1;
              }
              total_confidence += confidence;
              loss += confidence;
          }

          double other_temp;
          cudaDeviceSynchronize();
          stat = cublasDdot(handle, factors, r, 1, Xu, 1, &other_temp);
          if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("ddot 2 failed\n");
              cudaFree(YtY);
              cudaFree(r);
              return -1;
          }
          loss += other_temp;
          cudaDeviceSynchronize();
          stat = cublasDdot(handle, factors, Xu, 1, Xu, 1, &other_temp);
          if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("ddot 3 failed\n");
              cudaFree(YtY);
              cudaFree(r);
              return -1;
          }
          user_norm += other_temp;

          cudaFree(r);
      }

      for (int i = 0; i < items; ++i) {
          cudaDeviceSynchronize();
          double* Yi = &Y[i*factors];

          double other_temp;

          stat = cublasDdot(handle, factors, Yi, 1, Yi, 1, &other_temp);
          if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("ddot 4 failed\n");
              cudaFree(YtY);
              return -1;
          }
          item_norm += other_temp;
      }

      loss += reg * (item_norm + user_norm);

      cudaFree(YtY);

      return loss / ((double) (total_confidence + users * items - nnz));
  }
