#ifndef RECOMMEND_H
#define RECOMMEND_H

#include <list>
#include <vector>


std::list<int> recommend(int userid, std::vector<std::vector< int>> user_items, std::vector<std::vector< int>> user_factors, std::vector<std::vector< int>> item_factors, int N);

#endif
