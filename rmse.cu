#include "rmse.h"
// #define M 6
// #define N 5
// #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

//Cui will look like this:
//  Cui[0] = indptr - pointer into indices/data showing where each row starts
//  Cui[1] = indices - which columns in that row exist
//  Cui[2] = data - what's in that column

double rmse(cublasHandle_t handle, double* user_factors, double* item_factors,
            int* rows, int* cols, double* ratings, int num_things, int factors) {
    double error = 0;
    cublasStatus_t stat;

    for (int k = 0; k < num_things; ++k) {
        printf("rmse iter %d\n", k);
        cudaDeviceSynchronize();
        int uid = rows[k];
        int iid = cols[k];
        double rating = ratings[k];
        printf("%d\n", uid);
        printf("%d\n", iid);
        printf("%f\n", rating);

        double* user = &user_factors[uid*factors];
        double* item = &item_factors[iid*factors];
        cudaDeviceSynchronize();

        // printf("%f\n", user[0]);
        // printf("%f\n", item[0]);

        double guess;

        stat = cublasDdot(handle, factors, user, 1, item, 1, &guess);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("ddot failed\n");
            printf("%d\n", stat);
            return -1;
        }
        error += std::pow((rating-guess), 2);
        printf("error %f\n", error);
    }
    return std::sqrt(error/num_things);
}
