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
        printf("starting for loop\n");
        printf("users %d i %d\n", users, i);
        cudaDeviceSynchronize();

        int rowStart = indptr[i];
        int rowEnd = indptr[i+1];;

        printf("rowstart %d\n", rowStart);
        printf("rowEnd %d\n", rowEnd);

        printf("merp\n");
        int cols[rowEnd-rowStart];
        printf("%d\n", indic[rowStart]);

        memcpy(cols, &indic[rowStart], (rowEnd-rowStart)*sizeof(int));
        // err = cudaMemcpy(cols, ptrptr, rowEnd-rowStart, cudaMemcpyDeviceToHost);
        // if (err != cudaSuccess) {
        //   printf("first copy %s\n", cudaGetErrorString(err));
        //   return;
        // }
        printf("%d\n", cols[0]);
        printf("done w first memcpy\n");
        double vals[rowEnd-rowStart];
        memcpy(vals, &data[rowStart], (rowEnd-rowStart)*sizeof(double));
        // err = cudaMemcpy(vals, &data[rowStart], rowEnd-rowStart, cudaMemcpyDeviceToHost);
        // if (err != cudaSuccess) {
        //   printf("first copy %s\n", cudaGetErrorString(err));
        //   return;
        // }
        printf("done with memcpys\n");

        double diagVec[items];
        std::fill_n(diagVec, items, 0);
        //cudaDeviceSynchronize();
        for(int iter = 0; iter < (rowEnd-rowStart); ++iter)
        {
            printf("loooop\n");
            printf("%d\n", cols[iter]);
            printf("%f\n", vals[iter]);
            diagVec[cols[iter]] = vals[iter];
        }
        printf("making device vec\n");
        double* diagVec_dev;

        err = cudaMallocManaged(&diagVec_dev, items*sizeof(double));
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          return;
        }
        err = cudaMemcpy(diagVec_dev, diagVec, items, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          return;
        }
        printf("making ansC\n");

        double* ansC; //[items][factors];

        err = cudaMallocManaged(&ansC, items*factors*sizeof(double));
        if (err != cudaSuccess) {
          printf("%s\n", cudaGetErrorString(err));
          cudaFree(yt);
          cudaFree(diagVec_dev);
          cudaFree(ansC);
          return;
        }

        printf("dgmm\n");
        stat = cublasDdgmm(handle, mode, items, factors, y, items, diagVec, 1, ansC, items);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("ddgmm failed\n");
            cudaFree(yt);
            cudaFree(diagVec_dev);
            cudaFree(ansC);
            return;
        }
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

        printf("gemm\n");
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, factors, factors,
                        items, &alpha, yt, factors, ansC, items, &beta, ytcy, factors);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("dgemm failed\n");
            cudaFree(yt);
            cudaFree(diagVec_dev);
            cudaFree(ansC);
            cudaFree(ytcy);
            return;
        }
        cudaDeviceSynchronize();
        printf("setting diagonal\n");
        // for (int k = 0; k < factors; ++k)
        // {
        //     printf("%d\n",k);
        //     printf("%f\n", ytcy[0]);
        //     ytcy[factors*k +k] += reg;
        // }
        dim3 threads(32);
        dim3 blocks(iDivUp(factors*factors, 32));
        shiftDiagonal<<<blocks, threads>>>(ytcy, reg, factors);
        printf("done w diagonal\n");
        cudaDeviceSynchronize();

        double* ytcu; //[factors];
        printf("kjdfjkfdjkfd\n");
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
        printf("gemv\n");
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
        printf("solver fun times\n");
        int Lwork;
        cusolverDnDgetrf_bufferSize(solve_handle, factors, factors, ytcy, factors, &Lwork);
        int* rfOut;
        double workspace[Lwork];
        cusolverDnDgetrf(solve_handle, factors, factors, ytcy, factors, workspace, NULL, rfOut);
        int* rsOut;
        cusolverDnDgetrs(solve_handle, CUBLAS_OP_N, factors, 1, ytcy, factors, NULL, ytcu, factors, rsOut);
        memcpy(&x[(i*factors)], ytcu, factors);
        //x[i] = ytcu;
        cudaFree(diagVec_dev);
        cudaFree(ansC);
        cudaFree(ytcy);
        cudaFree(ytcu);
    }
    printf("done\n");
    cudaFree(yt);
}
