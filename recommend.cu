#include "recommend.h"
#include<iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
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

int main(int argc, char** argv) {
    char* fname = argv[1];
    int iterations = strtol(argv[2], NULL, 10);
    int factors = strtol(argv[3], NULL, 10);

    int users = 0;
    int items = 0;

    printf("%s\n", fname);

    FILE* fp;
    char* line = NULL;
    ssize_t read;
    size_t len = 0;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;

    fp = fopen(fname, "r");
    printf("done with setup\n");

    while ((read = getline(&line, &len, fp)) != -1) {
        int l = strlen(line);
        if (line[l-2] == ':') {
            indptr.push_back(indices.size());
        }
        else {
            int col = strtol(strtok(line, ","), NULL, 10);
            double rating = strtol(strtok(NULL, ","), NULL, 10);

            indices.push_back(col);
            data.push_back(rating);

            if (col > users) {
                users = col;
            }
        }
    }
    fclose(fp);

    items = indptr.size();
    users += 1;

    printf("users: %d\n", users);
    printf("items: %d\n", items);
    printf("factors: %d\n", factors);

    return 0;

}

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

double rmse(double** user_factors, double** item_factors, int* rows, int* cols, double* ratings, int num_things, int factors) {
    double error = 0;
    cublasHandle_t handle;
    cublasStatus_t err;

    for (int k = 0; k < num_things; ++k) {
        int uid = rows[k];
        int iid = cols[k];
        double rating = ratings[k];

        double* user = user_factors[uid];
        double* item = item_factors[iid];

        double guess;

        err = cublasDdot(handle, factors, user, 1, item, 1, &guess);
        error += std::pow((rating-guess), 2);
    }
    return std::sqrt(error/num_things);
}
