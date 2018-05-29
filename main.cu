#include<iostream> 
#include "recommend.h"

using namespace std;
 
void print(const list<int>& s) {
	list<int>::const_iterator i;
	for( i = s.begin(); i != s.end(); ++i)
		cout << *i << " ";
	cout << endl;
}

int main()
{
    vector<vector <int>> t1({{1,2,3}});
    cout<<"Hello World \n";
    const list<int> ans = (recommend(1, t1, t1, t1, 1));
    print(ans);
    return 0;
}