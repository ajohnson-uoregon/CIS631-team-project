#include<iostream> 
#include "recommend.h"
#include "recommend.cu"

using namespace std;
 
void print(const list<int>& s) {
	list<int>::const_iterator i;
	for( i = s.begin(); i != s.end(); ++i)
		cout << *i << " ";
	cout << endl;
}

int main(int argc, char** argv) {
    char* fname = argv[1];
    int iterations = strtol(argv[2], NULL, 10);
    int factors = strtol(argv[3], NULL, 10);

    int users = 0;
    int items = 0;

    printf("%s\n", fname);

    FILE* fp;
    char* line = NULL;
    ssize_t read;
    size_t len = 0;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;

    fp = fopen(fname, "r");
    printf("done with setup\n");

    while ((read = getline(&line, &len, fp)) != -1) {
        int l = strlen(line);
        if (line[l-2] == ':') {
            indptr.push_back(indices.size());
        }
        else {
            int col = strtol(strtok(line, ","), NULL, 10);
            double rating = strtol(strtok(NULL, ","), NULL, 10);

            indices.push_back(col);
            data.push_back(rating);

            if (col > users) {
                users = col;
            }
        }
    }
    fclose(fp);

    items = indptr.size();
    users += 1;

    printf("users: %d\n", users);
    printf("items: %d\n", items);
    printf("factors: %d\n", factors);

    return 0;

}
