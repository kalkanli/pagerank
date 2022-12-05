#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <omp.h>


using namespace std;

int MAX_ITER = 100;
float EPSILON = pow(10, -6);
float ALPHA = 0.8;
int STR_LEN = 26;

int main(int argc, char *argv[])
{
int count = 20;
    if (argc > 2)
    {
        printf("Too many arguments.");
        return -1;
    }

    time_t here1 = time(NULL);

    unordered_map<string, int> page_indices;
    unordered_map<int, int> outer_links;
    vector<int> row_begin;
    vector<int> col_indices;
    vector<int> rank_index;
    vector<pair<string, string> > edges; 

    fstream File(argv[1]); // read from file.
    
    string line;
    while (getline(File, line))
    {
        string tail_node = line.substr(0, STR_LEN);
        string head_node = line.substr(STR_LEN+1, STR_LEN);
        page_indices.insert(make_pair(head_node, page_indices.size()));
        edges.push_back(make_pair(tail_node, head_node));

        // count --;
        // if(count == 0) break;
    }
    time_t here2 = time(NULL);
    
    int old_head_index = -1;
    for(int i=0; i<edges.size(); i++) {
        int head_index = page_indices.find(edges.at(i).second)->second;
        int tail_index = page_indices.insert(make_pair(edges.at(i).first, page_indices.size())).first->second;
        if(tail_index == head_index) continue;
        if(head_index != old_head_index) {
            old_head_index = head_index;
            row_begin.push_back(col_indices.size());
        }
        col_indices.push_back(tail_index);
        outer_links.insert(make_pair(tail_index, 0)).first->second++;
    }

    row_begin.push_back(col_indices.size());
    for(int i=row_begin.size(); i <= page_indices.size(); i++) {
        row_begin.push_back(col_indices.size());
    }

    time_t here3 = time(NULL);
    cout << here3 - here1 << endl;

    int iterations = 0;
    vector<float> r(page_indices.size(), 1);

    omp_set_num_threads(6);
    
    while(iterations < MAX_ITER) {
        vector<float> r_new(page_indices.size(), 0);
        float difference = 0;
        int pid;
        int i;
        #pragma omp parallel shared(row_begin, outer_links, col_indices, r_new, difference) private(pid, i)
        {
            #pragma omp for schedule(static, 10)
            for(i=0; i<row_begin.size()-1; i++) { 
                r_new.at(i) = 0;
                for(int j=row_begin.at(i); j<row_begin.at(i+1); j++) { // row
                    r_new.at(i) += (1.0 / outer_links.find(col_indices.at(j))->second) * r.at(col_indices.at(j));
                }
                r_new.at(i) = ALPHA * r_new.at(i) + 1 - ALPHA;
                difference += abs(r_new.at(i) - r.at(i));
            }
        }
        r = r_new;
        iterations ++;
        if(difference < EPSILON) break;
    }

    cout << iterations << endl;

    

    return 0;
}