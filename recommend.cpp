B#include "recommend.h"
#include<iostream>
// #include <cuda_runtime.h>
// #include "cublas_v2.h"
// #define M 6
// #define N 5
// #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1)) 

std::list<int> recommend(int userid, std::vector<std::vector<int>> user_items, 
	       		     std::vector<std::vector< int>> user_factors, 
			     std::vector<std::vector< int>> item_factors, 
			     			      int N)
{
  int users = user_items.size(), items = user_items[0].size(), factors = user_items[0].size();
  // idk what type vector is so double is just a filler
  std::vector<double> user;
  for(int i = 0; i < factors; ++i) {
    // what does petsc setValues for a vec (list(range(factors)), user_factors.getValues([userid], list(range(factors)) do??
  }
  //this is possibly just ints but idk
  // index into vector by userid [0]
  std::vector<double> liked;
  std::vector<double> scores;
  // what is item factors??
  std::cout << users << std::endl;
  std::list<int> ans = {3};
  
  return ans;
}
