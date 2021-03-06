#include "recommend.h"


std::vector<double> multHelp(std::vector<double> vect1, std::vector<double> vect2) {
  std::vector<double> output;
  std::transform(vect1.begin(), vect1.end(), vect2.begin(), output.begin(), std::multiplies<double>()); 
  return output;
}
 
std::vector<std::tuple<int,double>> recommend(int userid, std::vector<std::vector<int>> user_items, 
	       		     std::vector<std::vector<double>> user_factors, 
			     std::vector<std::vector<double>> item_factors, 
			     			      int n)
{
  int users = user_items.size(), items = user_items[0].size(), factors = user_items[0].size();
  std::vector<double> user;
  user = user_factors[userid];  

  std::vector<int> liked = user_items[userid];
  std::vector<std::vector<double>> scores;
  
  for (auto iter = item_factors.begin(); iter != item_factors.end(); ++iter){
    scores.push_back(multHelp(*iter, user));
  }

  // score should be a vector 
  std::vector<double> score_vec = *(scores.begin());
  for (auto scoreI = ++(scores.begin()); scoreI != scores.end(); ++scoreI) {
    std::transform((*scoreI).begin(), (*scoreI).end(), score_vec.begin(), score_vec.begin(), std::plus<double>());
  } 
  // this isnt used but it is the python code
  int count = n + liked.size();
  
  std::vector<std::tuple<int,double>> scores_np;
  
  // check if it should be < or <=
  for (int i =0; i <score_vec.size(); ++i) {
    double currScore = score_vec[i];
    for(auto iterL = liked.begin(); iterL != liked.end(); ++iterL) {
      if(currScore == *iterL)
      {
        scores_np.push_back(std::tuple<int, double>(i, currScore));
      }
    }
  }
  // now sort by the second thing in the list
  //std::cout << users << std::endl;
  std::sort(scores_np.begin(), scores_np.end(), [](std::tuple<int,double> const &t1, std::tuple<int,double> const &t2) {
      return std::get<1>(t1) > std::get<1>(t2);
    });
  
  
  return scores_np;
}
