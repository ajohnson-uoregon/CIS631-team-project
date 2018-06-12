#include<iostream>
#include "recommend.h"
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstring>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "rmse.h"
#include "leastsqr.h"
#include "calculateLoss.h"

void fileProcess(char* fname, std::vector<int>* indptr, std::vector<int>* indices,
                    std::vector<double>* data, int* users, int* items)
{
    FILE* fp;
    char* line = NULL;
    ssize_t read;
    size_t len = 0;
    fp = fopen(fname, "r");
        printf("done with setup\n");

        while ((read = getline(&line, &len, fp)) != -1) {
            int l = strlen(line);
            if (line[l-2] == ':') {
                indptr->push_back(indices->size());
            }
            else {
                int col = strtol(strtok(line, ","), NULL, 10);
                double rating = strtol(strtok(NULL, ","), NULL, 10);

                indices->push_back(col);
                data->push_back(rating);

                if (col > *users) {
                    *users = col;
                }
            }
        }

    *items = indptr->size();
    *users += 1;
    fclose(fp);
}

void fileDense(char* fname, std::vector<int>* rows, std::vector<int>* cols,
                    std::vector<double>* data, int* length)
{
    FILE* fp;
    char* line = NULL;
    ssize_t read;
    size_t len = 0;
    fp = fopen(fname, "r");
    printf("done with setup\n");
    int index = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        int l = strlen(line);
        int col;
        if (line[l-2] == ':') {
            col = strtol(strtok(line, ":"), NULL, 10);
        }
        else {
            int row = strtol(strtok(line, ","), NULL, 10);
            double rating = strtol(strtok(NULL, ","), NULL, 10);

            cols->push_back(col);
            rows->push_back(row);
            data->push_back(rating);
        }
    }

    *length = rows->size();
    fclose(fp);
}



int main(int argc, char** argv) {
    std::cout << "here"<< std::endl;
    char* fname = argv[1];
    char* fnameT = argv[2];
    char* fnameD = argv[3];
    int iterations = strtol(argv[4], NULL, 10);
    int factors = strtol(argv[5], NULL, 10);

    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaError_t err;

    int users = 0;
    int items = 0;

    int usersT = 0;
    int itemsT = 0;

    printf("%s\n", fname);


    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;

    std::vector<int> indptrT;
    std::vector<int> indicesT;
    std::vector<double> dataT;

    //need to do file processing twice
    fileProcess(fname, &indptr, &indices, &data, &users, &items);
    fileProcess(fnameT,&indptrT, &indicesT, &dataT, &usersT, &itemsT);

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    //create factors matrices x2
    // double userFactors[users][factors];
    // double itemFactors[items][factors];

    double* userFactors;
    double* itemFactors;

    err = cudaMallocManaged(&userFactors, users*factors*sizeof(double));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      return;
    }

    err = cudaMallocManaged(&itemFactors, items*factors*sizeof(double));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      return;
    }
    GPU_fill_rand(userFactors, users, factors);
    GPU_fill_rand(itemFactors, items, factors);

    //run iterations of leastsqrs and calculateLoss
    for(int i=0; i <iterations; ++i) {
        least_squares(handle, indptr.data(), indices.data(), data.data(), users,
                        items, factors, userFactors[0], itemFactors[0],
                        .01, 0, users);
        least_squares(handle, indptrT.data(), indicesT.data(), dataT.data(), items,
                        users, factors, itemFactors[0], userFactors[0],
                        .01, 0, items);
        calculate_loss(handle, indptr.data(), indices.data(), data.data(), userFactors[0], itemFactors[0], .01,
                        users, items, factors, data.size());
    }
    // run rmse
    double totErr;
    //come back to
    std::vector<int> testRow;
    std::vector<int> testCol;
    std::vector<double> testData;
    int testLength;
    fileDense(fnameD, &testRow, &testCol, &testData, &testLength);
    totErr = rmse(handle, userFactors[0], itemFactors[0], testRow.data(),
                testCol.data(), testData.data(),
                testLength, factors);
    // we guchhi
    std::cout<<totErr<<std::endl;
    // predict things and see accuracy


    printf("users: %d\n", users);
    printf("items: %d\n", items);
    printf("factors: %d\n", factors);

    return 0;

}
