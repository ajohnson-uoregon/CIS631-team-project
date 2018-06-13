#include "calculateLoss.h"

double calculate_loss(cublasHandle_t handle, int* indptr, int* indices, double* data, double* X, double* Y, double reg,
    int users, int items, int factors, int nnz) {

      int loss = 0;
      int total_confidence = 0;
      int item_norm = 0;
      int user_norm = 0;
      cublasStatus_t err;

      // malloc this
      double** YtY;
      const double alpha = 1;
      const double beta = 0;

      // do transpose
      err = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, items, factors,
                &alpha, Y, items, &beta, Y, items, YtY[0], factors);


      for (int u = 0; u < users; ++u) {
          cudaDeviceSynchronize();
          double temp = 1.0;

          double* r;
          double* Xu = &X[u*factors];


          err = cublasDgemv(handle, CUBLAS_OP_N, items, factors,
                    &alpha, Y, items, Xu, 1, &beta, r, 1);
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
              err = cublasDdot(handle, factors, Yi, 1, Xu, 1, &d);
              temp = (confidence - 1)*d - (2*confidence);
              cudaDeviceSynchronize();
              err = cublasDaxpy(handle, factors, &temp, Yi, 1, r, 1);
              total_confidence += confidence;
              loss += confidence;
          }

          double other_temp;
          cudaDeviceSynchronize();
          err = cublasDdot(handle, factors, r, 1, Xu, 1, &other_temp);
          loss += other_temp;
          cudaDeviceSynchronize();
          err = cublasDdot(handle, factors, Xu, 1, Xu, 1, &other_temp);
          user_norm += other_temp;
      }

      for (int i = 0; i < items; ++i) {
          cudaDeviceSynchronize();
          double* Yi = &Y[i*factors];

          double other_temp;

          err = cublasDdot(handle, factors, Yi, 1, Yi, 1, &other_temp);
          item_norm += other_temp;
      }

      loss += reg * (item_norm + user_norm);

      return loss / (total_confidence + users * items - nnz);
  }
