#include "recommend.h"
#include<iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
// #define M 6
// #define N 5
// #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1)) 

std::list<int> recommend(int userid, std::vector<std::vector<int>> user_items, std::vector<std::vector< int>> user_factors, std::vector<std::vector< int>> item_factors, int N)
{
    int users = user_items.size();
    int items = user_items[0].size();
    int factors = user_items[0].size();
    std::cout << users << std::endl;
    std::list<int> ans = {3};
    return ans;
}