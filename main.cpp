#include<iostream> 
#include "recommend.h"
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstring>
void fileProcess(char* fname, std::vector<int>* indptr, std::vector<int>* indices.
                    std::vector<double>* data)
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
}

int main(int argc, char** argv) {
    std::cout << "here"<< std::endl;  
    char* fname = argv[1];
    char* fnameT = argv[2]];
    int iterations = strtol(argv[3], NULL, 10);
    int factors = strtol(argv[4], NULL, 10);

    int users = 0;
    int items = 0;

    printf("%s\n", fname);

    
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;

    std::vector<int> indptrT;
    std::vector<int> indicesT;
    std::vector<double> dataT;

    //need to do file processing twice
    fileProcess(fname, &indptr, &indices, &data);
    fileProcess(fnameT,&indptrT, &indicesT, &dataT);
    //create factors matrices x2
    double userFactors[users][factors];
    double itemFactors[items][factors];

    GPU_fill_rand(userFactors, users, factors);
    GPU_fill_rand(itemFactors, items, factors);

    //run iterations of leastsqrs and calculateLoss
    for(int i=0; i <iterations; ++i) {
        least_squares(indptr.data(), indices.data(), data.data(),
                        items, factors, userFactors, itemFactors,
                        .01, 0, users);
        least_squares(indptrT.data(), indicesT.data(), dataT.data(),
                        users, factors, itemFactors, userFactors,
                        .01, 0, items);
        calculate_loss(indptr.data(), userFactors, itemFactors, .01, 
                        users, items, factors, data.size());
    }
    // run rmse
    double totErr;
    //come back to
    //totErr = rmse(user_factors, item_factors, )
    // we guchhi
    // predict things and see accuracy 

    items = indptr.size();
    users += 1;

    printf("users: %d\n", users);
    printf("items: %d\n", items);
    printf("factors: %d\n", factors);

    return 0;

}
