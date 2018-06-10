#include "recommend.h"
#include<iostream>
#include <algorithm>
#include <accumulate>
// #include <cuda_runtime.h>
// #include "cublas_v2.h"
// #define M 6
// #define N 5
// #define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1)) 

//add to the header file
std::vector<double> multHelp(std::vector<double> vect1, std::vector<double> vect2) {
  std::vector<double> output;
  // foreach maybe replace
  std::transform(vect1.begin(), vect1.end(), vect2.begin(), output.begin(), std::multiplies<double>()); 
  return output;
}

std::list<int> recommend(int userid, std::vector<std::vector<int>> user_items, 
	       		     std::vector<std::vector<double>> user_factors, 
			     std::vector<std::vector<double>> item_factors, 
			     			      int n)
{
  int users = user_items.size(), items = user_items[0].size(), factors = user_items[0].size();
  // idk what type vector is so double is just a filler
  std::vector<double> user;
  user = user_factors[userid];  
  //this is possibly just ints but idk
  // index into vector by userid [0]
  std::vector<int> liked = user_items[userid];
  std::vector<std::vector<double>> scores;
  // what is item factors??
  // item_factors is a matrix * user vector
  // for vector in item_factors call mult help
 
  // std::transform(item_factors.begin(), item_factors.end(), , std::bind1st(std::multiplies<T>(),3));
  for (auto iter = item_factors.begin(); iter != item_factors.end(); ++iter){
    scores.push_back(multHelp(&iter, user))
  }

  // idk what scores_np = scores.getArray() does so I am ignoring it for now
  // transform on outer and accumulate on inner to sum score
  int count = n + liked.size();
  
  std::vector<std::vector<double>> scores_np;
  // check if it should be < or <=
  for (int i =0; i <scores.size(); i++) {
    currScore = scores[i];
    for(auto iterL = liked.begin(); iterL != liked.end(); ++iterL) {
      if(currScore == &iterL)
      {
        scores_np.push_back(currScore);
      }
    }
  }


  //std::cout << users << std::endl;
  std::list<int> ans = {3};
  
  return ans;
}
