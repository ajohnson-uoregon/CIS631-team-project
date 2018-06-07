#ifndef RECOMMEND_H
#define RECOMMEND_H

#include <list>
#include <vector>

std::vector<double> multHelp(std::vector<double> vect1, std::vector<double> vect2);
std::list<int> recommend(int userid, std::vector<std::vector<int> > user_items, std::vector<std::vector<double> > user_factors, std::vector<std::vector< double> > item_factors, int n);

#endif
