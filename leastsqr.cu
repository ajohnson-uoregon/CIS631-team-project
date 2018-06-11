#include <cstring>
#include <algoithm>
double** least_squares (int* indptr, int* indic, double* data, int users, 
                        int items, int factors,
                        double** x, double** y, 
                        double reg, int istart, int iend) 
{
    cublasStatus_t err;
    cublasHandle_t handle;
    const double alpha = 1;
    const double beta = 0;
    double[factors][items] yt;
    err = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, items, factors,
        &alpha, y[0], items, &beta, y[0], items, yt[0], factors);
    
    cublasSideMode_t mode = CUBLAS_SIDE_LEFT;

    for (int i =0; i < users; ++i) {
        
        int rowStart = indptr[i];
        int rowEnd = intptr[i+1];

        int cols[rowEnd-rowStart];
        memcpy(cols, &indic[rowStart], rowEnd-rowStart);
        
        double vals[rowEnd-rowStart];
        memcpy(vals, &data[rowStart], rowEnd-rowStart);

        double[items] diagVec;
        fill(diagVec.start(), diagVec.end(), 0);

        for(int iter = 0; iter < (rowEnd-rowStart); ++iter) 
        {
            diagVec[cols[iter]] = vals[iter];
        }   
        double[items][factors] ansC;

        cublasDdgmm(handle, mode, items, factors, y[0], items, diagVec, 1, ansC, items);
        double[factors][factors] ytcy;

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, factors, factors, 
                        items, &alpha, yt, factors, ansC, items, &beta, ytcy, factors);
        for (int i = 0; i < factors; ++i) 
        {
            ytcy[i][i] += reg;
        }

        double[factors] ytcu;
        cublasDgemv (handle, CUBLAS_OP_N, CUBLAS_OP_N, factors, items,
                        &alpha, yt, factors, diagVec, 1, &beta, ytcu, 1);
        int* Lwork;
        cusolverDnDgetrf_bufferSize(handle, factors, factors, ytcy, facotrs, Lwork);
        int* rfOut;
        cusolverDnDgetrf(handle, factors, factors, ytcy, factors, Lwork, NULL, rfOut);
        int* rsOut;
        cusolverDnDgetrs(handle, CUBLAS_OP_N, factors, 1, ytcy, factors, NULL, ytcu, factors, rsOut);
        x[i] = ytcu;
    }

    return x;

}