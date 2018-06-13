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
    indptr->push_back(indices->size());
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


    std::vector<int> indptr_vec;
    std::vector<int> indices_vec;
    std::vector<double> data_vec;

    std::vector<int> indptrT_vec;
    std::vector<int> indicesT_vec;
    std::vector<double> dataT_vec;

    //need to do file processing twice
    fileProcess(fname, &indptr_vec, &indices_vec, &data_vec, &users, &items);
    fileProcess(fnameT,&indptrT_vec, &indicesT_vec, &dataT_vec, &usersT, &itemsT);

    printf("done reading from files\n");

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        printf("%d\n", stat);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    printf("done initializing cublas\n");
    //create factors matrices x2
    // double userFactors[users][factors];
    // double itemFactors[items][factors];

    double* userFactors;
    double* itemFactors;

    err = cudaMallocManaged(&userFactors, users*factors*sizeof(double));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cublasDestroy(handle);
      return -1;
    }

    printf("malloced userfactors\n");

    err = cudaMallocManaged(&itemFactors, items*factors*sizeof(double));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cublasDestroy(handle);
      return -1;
    }

    printf("malloced itemfactors\n");

    GPU_fill_rand(userFactors, users, factors);
    GPU_fill_rand(itemFactors, items, factors);

    printf("filled random matrices\n");

    int* indptr;
    int* indices;
    double* data;

    int* indptrT;
    int* indicesT;
    double* dataT;

    err = cudaMallocManaged(&indptr, indptr_vec.size()*sizeof(int));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cublasDestroy(handle);
      return -1;
    }

    printf("copying values into indptr\n");
    err = cudaMemcpy(indptr, indptr_vec.data(), indptr_vec.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cublasDestroy(handle);
      return -1;
    }

    err = cudaMallocManaged(&indices, indices_vec.size()*sizeof(int));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cudaFree(indices);
      cublasDestroy(handle);
      return -1;
    }

    printf("copying values into indices\n");
    err = cudaMemcpy(indices, indices_vec.data(), indices_vec.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cudaFree(indices);
      cublasDestroy(handle);
      return -1;
    }

    err = cudaMallocManaged(&data, data_vec.size()*sizeof(double));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cudaFree(indices);
      cudaFree(data);
      cublasDestroy(handle);
      return -1;
    }

    printf("copying values into data\n");
    err = cudaMemcpy(data, data_vec.data(), data_vec.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cudaFree(indices);
      cudaFree(data);
      cublasDestroy(handle);
      return -1;
    }

    //////////////////////

    err = cudaMallocManaged(&indptrT, indptrT_vec.size()*sizeof(int));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cudaFree(indices);
      cudaFree(data);
      cudaFree(indptrT);
      cublasDestroy(handle);
      return -1;
    }

    printf("copying values into indptrT\n");
    err = cudaMemcpy(indptrT, indptrT_vec.data(), indptrT_vec.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cudaFree(indices);
      cudaFree(data);
      cudaFree(indptrT);
      cublasDestroy(handle);
      return -1;
    }

    err = cudaMallocManaged(&indicesT, indicesT_vec.size()*sizeof(int));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cudaFree(indices);
      cudaFree(data);
      cudaFree(indptrT);
      cudaFree(indicesT);
      cublasDestroy(handle);
      return -1;
    }

    printf("copying values into indicesT\n");
    err = cudaMemcpy(indicesT, indicesT_vec.data(), indicesT_vec.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cudaFree(indices);
      cudaFree(data);
      cudaFree(indptrT);
      cudaFree(indicesT);
      cublasDestroy(handle);
      return -1;
    }

    err = cudaMallocManaged(&dataT, dataT_vec.size()*sizeof(double));
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cudaFree(indices);
      cudaFree(data);
      cudaFree(indptrT);
      cudaFree(indicesT);
      cudaFree(dataT);
      cublasDestroy(handle);
      return -1;
    }

    printf("copying values into dataT\n");
    err = cudaMemcpy(dataT, dataT_vec.data(), dataT_vec.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      cudaFree(userFactors);
      cudaFree(itemFactors);
      cudaFree(indptr);
      cudaFree(indices);
      cudaFree(data);
      cudaFree(indptrT);
      cudaFree(indicesT);
      cudaFree(dataT);
      cublasDestroy(handle);
      return -1;
    }
    cudaDeviceSynchronize();
    for (int i =0; i < 10; ++i) {
      printf("%f\n", userFactors[i]);
      printf("%f\n", data_vec[i]);
      printf("%f\n", data[i]);
      printf("%d\n", indptr[i]);
      printf("%d\n", indices[i]);
      printf("\n");
    }
    //printf("%d\n", indptr_vec.data()[0]);
    //printf("%d\n", indptr_vec[0]);
    //printf("%d\n", indptr_vec[1]);

    //run iterations of leastsqrs and calculateLoss
    for(int i=0; i <iterations; ++i) {
        printf("first least squares iter %d\n", i);
        least_squares(handle, indptr, indices, data, users,
                        items, factors, userFactors, itemFactors,
                        .01, 0, users);
        printf("second least squares iter %d\n", i);
        least_squares(handle, indptrT, indicesT, dataT, items,
                        users, factors, itemFactors, userFactors,
                        .01, 0, items);
        printf("calculate loss iter %d\n", i);
        calculate_loss(handle, indptr, indices, data,
                        userFactors, itemFactors, .01,
                        users, items, factors, data_vec.size());
    }

    printf("freeing things\n");
    cudaFree(userFactors);
    cudaFree(itemFactors);
    cudaFree(indptr);
    cudaFree(indices);
    cudaFree(data);
    cudaFree(indptrT);
    cudaFree(indicesT);
    cudaFree(dataT);
    cublasDestroy(handle);
    return 0;
    // run rmse
    double totErr;
    //come back to
    std::vector<int> testRow;
    std::vector<int> testCol;
    std::vector<double> testData;
    int testLength;
    fileDense(fnameD, &testRow, &testCol, &testData, &testLength);
    totErr = rmse(handle, userFactors, itemFactors, testRow.data(),
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
