#include "leastsqr.h"

void GPU_fill_rand(double *A, const int nr_rows_A, const int nr_cols_A)
{
    curandGenerator_t prng;
    curandStatus_t stat;

    stat = curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    if (stat != CURAND_STATUS_SUCCESS) {
        printf("error at %s:%d\n",__FILE__, __LINE__);
        return;
    }
    stat = curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    if (stat != CURAND_STATUS_SUCCESS) {
        printf("error at %s:%d\n",__FILE__, __LINE__);
        return;
    }

    stat = curandGenerateUniformDouble(prng, A, nr_rows_A * nr_cols_A);
    if (stat != CURAND_STATUS_SUCCESS) {
        printf("error at %s:%d\n",__FILE__, __LINE__);
        return;
    }

}
// borrowed from https://stackoverflow.com/questions/21112725/cuda-fill-an-matrix-with-random-values-between-a-and-b
int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

__global__ void shiftDiagonal(double *A, double reg, int num_cols) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_cols) {
      A[num_cols*tid + tid] += reg;
    }
}


void least_squares (cublasHandle_t handle, int* indptr, int* indic, double* data, int users,
                        int items, int factors,
                        double* x, double* y,
                        double reg, int istart, int iend)
{
    cudaError_t err;
    cublasStatus_t stat;
    cusolverDnHandle_t solve_handle;
    cusolverStatus_t solve_stat;
    const double alpha = 1;
    const double beta = 0;

    //double yt[factors][items];

    double* yt;

    err = cudaMallocManaged(&yt, factors*items*sizeof(double));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(yt);
      return;
    }

    stat = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, factors, items,
        &alpha, y, items, &beta, y, items, yt, factors);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("dgeam failed\n");
        cudaFree(yt);
        return;
    }

    cublasSideMode_t mode = CUBLAS_SIDE_LEFT;

    for (int i =0; i < users; ++i) {
      if (i % 10000 == 0 || i == users-1) {
        printf("iteration %d\n", i);
      }
        //printf("starting for loop\n");
        // printf("users %d i %d\n", users, i);
        cudaDeviceSynchronize();

        int rowStart = indptr[i];
        int rowEnd = indptr[i+1];

        if (rowStart == rowEnd) {
          printf("AW CRAP\n");
        }

        // printf("rowstart %d\n", rowStart);
        // printf("rowEnd %d\n", rowEnd);

        // printf("merp\n");
        int cols[rowEnd-rowStart];
        // printf("%d\n", indic[rowStart]);

        memcpy(cols, &indic[rowStart], (rowEnd-rowStart)*sizeof(int));
        // err = cudaMemcpy(cols, ptrptr, rowEnd-rowStart, cudaMemcpyDeviceToHost);
        // if (err != cudaSuccess) {
        //   printf("first copy %s\n", cudaGetErrorString(err));
        //   return;
        // }
        // printf("%d\n", cols[0]);
        // printf("done w first memcpy\n");
        double vals[rowEnd-rowStart];
        memcpy(vals, &data[rowStart], (rowEnd-rowStart)*sizeof(double));
        // err = cudaMemcpy(vals, &data[rowStart], rowEnd-rowStart, cudaMemcpyDeviceToHost);
        // if (err != cudaSuccess) {
        //   printf("first copy %s\n", cudaGetErrorString(err));
        //   return;
        // }
        // printf("done with memcpys\n");

        double diagVec[items];
        std::fill_n(diagVec, items, 0);
        // printf("items %d\n", items);
        //cudaDeviceSynchronize();
        for(int iter = 0; iter < (rowEnd-rowStart); ++iter)
        {
            // printf("loooop\n");
            // printf("%d\n", cols[iter]);
            // printf("%f\n", vals[iter]);
            //printf("test\n");
            diagVec[cols[iter]] = vals[iter];
            //printf("success\n");
        }
        // printf("making device vec\n");
        double* diagVec_dev;
        cudaDeviceSynchronize();
        err = cudaMallocManaged(&diagVec_dev, items*sizeof(double));
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          return;
        }
        err = cudaMemcpy(diagVec_dev, diagVec, items*sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          return;
        }
        // printf("making ansC\n");

        double* ansC; //[items][factors];

        err = cudaMallocManaged(&ansC, items*factors*sizeof(double));
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          return;
        }
        cudaDeviceSynchronize();
        // printf("dgmm\n");
        stat = cublasDdgmm(handle, mode, items, factors, y, items, diagVec_dev, 1, ansC, items);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("ddgmm failed\n");
            cudaFree(yt);
            cudaFree(diagVec_dev);
            cudaFree(ansC);
            return;
        }
        cudaDeviceSynchronize();
        double* ytcy; //[factors][factors];

        err = cudaMallocManaged(&ytcy, factors*factors*sizeof(double));
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          cudaFree(ytcy);
          return;
        }

        // printf("gemm\n");
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, factors, factors,
                        items, &alpha, yt, factors, ansC, items, &beta, ytcy, factors);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("dgemm failed\n");
            // printf("%d\n", stat);
            cudaFree(yt);
            cudaFree(diagVec_dev);
            cudaFree(ansC);
            cudaFree(ytcy);
            return;
        }
        cudaDeviceSynchronize();
        // printf("setting diagonal\n");
        // for (int k = 0; k < factors; ++k)
        // {
        //     printf("%d\n",k);
        //     printf("%f\n", ytcy[0]);
        //     ytcy[factors*k +k] += reg;
        // }
        // dim3 threads(32);
        // dim3 blocks(iDivUp(factors*factors, 32));
        // shiftDiagonal<<<blocks, threads>>>(ytcy, reg, factors);
        // printf("done w diagonal\n");
        //cudaDeviceSynchronize();

        double* ytcu; //[factors];
        // printf("kjdfjkfdjkfd\n");
        err = cudaMallocManaged(&ytcu, factors*sizeof(double));
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          cudaFree(ytcy);
          cudaFree(ytcu);
          return;
        }
        cudaDeviceSynchronize();
        // printf("gemv\n");
        stat = cublasDgemv (handle, CUBLAS_OP_N, factors, items,
                        &alpha, yt, factors, diagVec_dev, 1, &beta, ytcu, 1);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("dgemm failed\n");
            cudaFree(yt);
            cudaFree(diagVec_dev);
            cudaFree(ansC);
            cudaFree(ytcy);
            cudaFree(ytcu);
            return;
        }
        cudaDeviceSynchronize();
        // printf("solver fun times\n");
        solve_stat = cusolverDnCreate(&solve_handle);
        if (CUSOLVER_STATUS_SUCCESS != solve_stat) {
          printf("creating solver failed\n");
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          cudaFree(ytcy);
          cudaFree(ytcu);
          return;
        }
        cudaDeviceSynchronize();
        int Lwork;
        // printf("buffersize\n");
        solve_stat = cusolverDnDpotrf_bufferSize(solve_handle, CUBLAS_FILL_MODE_UPPER,
          factors, ytcy, factors, &Lwork);
        if (CUSOLVER_STATUS_SUCCESS != solve_stat) {
          printf("get buffer size failed\n");
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          cudaFree(ytcy);
          cudaFree(ytcu);
          return;
        }
        cudaDeviceSynchronize();
        int* rfOut;
        err = cudaMallocManaged(&rfOut, sizeof(int));
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          cudaFree(ytcy);
          cudaFree(ytcu);
          cudaFree(rfOut);
          return;
        }
        double* workspace;
        err = cudaMallocManaged(&workspace, Lwork*sizeof(double));
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          cudaFree(ytcy);
          cudaFree(ytcu);
          cudaFree(rfOut);
          cudaFree(workspace);
          return;
        }
        cudaDeviceSynchronize();
        // printf("factorizing\n");
        solve_stat = cusolverDnDpotrf(solve_handle, CUBLAS_FILL_MODE_UPPER,
           factors, ytcy, factors, workspace, Lwork, rfOut);
        if (CUSOLVER_STATUS_SUCCESS != solve_stat) {
          printf("factorization failed\n");
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          cudaFree(ytcy);
          cudaFree(ytcu);
          cudaFree(rfOut);
          cudaFree(workspace);
          return;
        }
        //printf("%d\n", *rfOut);
        cudaDeviceSynchronize();
        int* rsOut;
        err = cudaMallocManaged(&rsOut, sizeof(int));
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          cudaFree(ytcy);
          cudaFree(ytcu);
          cudaFree(rfOut);
          cudaFree(workspace);
          cudaFree(rsOut);
          return;
        }
        // printf("solving\n");
        solve_stat = cusolverDnDpotrs(solve_handle, CUBLAS_FILL_MODE_UPPER,
          factors, 1, ytcy, factors, ytcu, factors, rsOut);
        if (CUSOLVER_STATUS_SUCCESS != solve_stat) {
          printf("solver failed\n");
          printf("err %d\n", solve_stat);
          // printf("CUSOLVER_STATUS_SUCCESS = %d \n", CUSOLVER_STATUS_SUCCESS);
          // printf("CUSOLVER_STATUS_NOT_INITIALIZED = %d \n", CUSOLVER_STATUS_NOT_INITIALIZED);
          // printf("CUSOLVER_STATUS_INVALID_VALUE = %d \n", CUSOLVER_STATUS_INVALID_VALUE);
          // printf("CUSOLVER_STATUS_ARCH_MISMATCH = %d \n", CUSOLVER_STATUS_ARCH_MISMATCH);
          // printf("CUSOLVER_STATUS_INTERNAL_ERROR = %d \n", CUSOLVER_STATUS_INTERNAL_ERROR);
          printf("out %d\n", *rsOut);
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          cudaFree(ytcy);
          cudaFree(ytcu);
          cudaFree(rfOut);
          cudaFree(workspace);
          cudaFree(rsOut);
          return;
        }
        cudaDeviceSynchronize();
        // printf("doing memcpy\n");
        cudaMemcpy(&x[i*factors], &ytcu[0], factors*sizeof(double), cudaMemcpyDeviceToDevice);
        //x[i] = ytcu;
        cudaDeviceSynchronize();
        // printf("freeing things\n");
        cusolverDnDestroy(solve_handle);
        cudaFree(diagVec_dev);
        cudaFree(ansC);
        cudaFree(ytcy);
        cudaFree(ytcu);
        cudaFree(rfOut);
        cudaFree(workspace);
        cudaFree(rsOut);
    }
    printf("done\n");
    cudaFree(yt);
}
