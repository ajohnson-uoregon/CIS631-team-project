#include "rmse.h"
// #define M 6
// #define N 5
// #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1)) 

//Cui will look like this:
//  Cui[0] = indptr - pointer into indices/data showing where each row starts
//  Cui[1] = indices - which columns in that row exist
//  Cui[2] = data - what's in that column

double rmse(double* user_factors, double* item_factors, int* rows, int* cols, double* ratings, int num_things, int factors) {
    double error = 0;
    cublasHandle_t handle;
    cublasStatus_t err;

    for (int k = 0; k < num_things; ++k) {
        int uid = rows[k];
        int iid = cols[k];
        double rating = ratings[k];

        double* user = &user_factors[uid*k];
        double* item = &item_factors[iid*k];

        double guess;

        err = cublasDdot(handle, factors, user, 1, item, 1, &guess);
        error += std::pow((rating-guess), 2);
    }
    return std::sqrt(error/num_things);
}
