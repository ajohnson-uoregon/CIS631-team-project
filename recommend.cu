#include "recommend.h"
#include<iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
// #define M 6
// #define N 5
// #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1)) 
/*
std::list<int> recommend(int userid, std::vector<std::vector<int> > user_items, std::vector<std::vector<int> > user_factors, std::vector<std::vector<int> > item_factors, int N)
{
    int users = user_items.size();
    int items = user_items[0].size();
    int factors = user_items[0].size();
    std::cout << users << std::endl;
    std::list<int> ans = {3};
    return ans;
}
*/
//Cui will look like this:
//  Cui[0] = indptr - pointer into indices/data showing where each row starts
//  Cui[1] = indices - which columns in that row exist
//  Cui[2] = data - what's in that column

double calculate_loss(double** Cui, double** X, double** Y, double reg, 
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
        
        int rowStart = Cui[0][u];
        int rowEnd = Cui[0][u+1];

        int cols[rowEnd-rowStart];
        memcpy(cols, &Cui[1][rowStart], rowEnd-rowStart);
        //int* cols = Cui[1][rowStart:rowEnd];
        
        double vals[rowEnd-rowStart];
        memcpy(vals, &Cui[2][rowStart], rowEnd-rowStart);
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
/*
def rmse(values, user_factors, item_factors):
    error = 0
    indptr, indices, data = values
    num_things = 0
    for uid in range(len(indptr) - 2):
        row_start = indptr[uid]
        row_end = indptr[uid+1]
        for index in range(row_start,row_end):
            iid = indices[index]
            rating = data[index]

            user = petsc.Vec()
            user.createSeq(factors)
            user.setValues(list(range(factors)), user_factors.getValues([uid], list(range(factors))))
            user.assemble()

            item = petsc.Vec()
            item.createSeq(factors)
            item.setValues(list(range(factors)), item_factors.getValues([iid], list(range(factors))))
            item.assemble()

            guess = user.dot(item)

            error += (rating-guess)**2
            num_things += 1

    return math.sqrt(error/num_things)
*/
