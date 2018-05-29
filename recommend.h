#ifndef RECOMMEND_H
#define RECOMMEND_H

#include <list>
#include <vector>

using namespace std;


list<int> recommend(int userid, vector<vector< int>> user_items, vector<vector< int>> user_factors, vector<vector< int>> item_factors, int N);

#endif