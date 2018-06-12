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


void least_squares (cublasHandle_t handle, int* indptr, int* indic, double* data, int users, 
                        int items, int factors,
                        double* x, double* y, 
                        double reg, int istart, int iend) 
{
    cublasStatus_t err;
    cusolverDnHandle_t solve_handle;
    const double alpha = 1;
    const double beta = 0;
    double yt[factors][items];
    
    err = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, items, factors,
        &alpha, y, items, &beta, y, items, yt[0], factors);
    
    cublasSideMode_t mode = CUBLAS_SIDE_LEFT;

    for (int i =0; i < users; ++i) {
        
        int rowStart = indptr[i];
        int rowEnd = indptr[i+1];

        int cols[rowEnd-rowStart];
        memcpy(cols, &indic[rowStart], rowEnd-rowStart);
        
        double vals[rowEnd-rowStart];
        memcpy(vals, &data[rowStart], rowEnd-rowStart);

        double diagVec[items];
        std::fill_n(diagVec, items, 0);

        for(int iter = 0; iter < (rowEnd-rowStart); ++iter) 
        {
            diagVec[cols[iter]] = vals[iter];
        }   
        double ansC[items][factors];

        cublasDdgmm(handle, mode, items, factors, y, items, diagVec, 1, ansC[0], items);
        double ytcy[factors][factors];

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, factors, factors, 
                        items, &alpha, yt[0], factors, ansC[0], items, &beta, ytcy[0], factors);
        for (int k = 0; k < factors; ++k) 
        {
            ytcy[k][k] += reg;
        }

        double ytcu[factors];
        cublasDgemv (handle, CUBLAS_OP_N, factors, items,
                        &alpha, yt[0], factors, diagVec, 1, &beta, ytcu, 1);
        int Lwork;
        cusolverDnDgetrf_bufferSize(solve_handle, factors, factors, ytcy[0], factors, &Lwork);
        int* rfOut;
        double workspace[Lwork];
        cusolverDnDgetrf(solve_handle, factors, factors, ytcy[0], factors, workspace, NULL, rfOut);
        int* rsOut;
        cusolverDnDgetrs(solve_handle, CUBLAS_OP_N, factors, 1, ytcy[0], factors, NULL, ytcu, factors, rsOut);
        memcpy(&x[(i*factors)], ytcu, factors);
        //x[i] = ytcu;
    }
}
