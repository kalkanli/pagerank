#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <omp.h>

using namespace std;

int MAX_ITER = 100;                         // maximum number of iterations before stopping.
float EPSILON = pow(10, -6);                // tolerance for stopping the iterations.
float ALPHA = 0.8;                          // damping factor.
int STR_LEN = 26;                           // length of the domain ids in the file.

unordered_map<int, string> page_ids;        // i => node map
unordered_map<string, int> page_indices;    // node => i map
unordered_map<int, int> outer_links;        // # of outgoing edges from the ith node
vector<int> row_begin;                      // stores the row beginning for CSR representation. for zero rows same value repeated.
vector<int> col_indices;                    // ith element of this list is the column index of ith element in values (only 1s) array.

int load_file_to_csr_matrix(char * filename);

int main(int argc, char *argv[])
{
    int count = 20;
    if (argc > 2)
    {
        printf("Too many arguments.");
        return -1;
    }

    time_t before_load = time(NULL);

    load_file_to_csr_matrix(argv[1]);

    time_t after_load = time(NULL);

    int iterations = 0;
    vector<float> r(page_indices.size(), 1); // form a rank vector with 1s.
    omp_set_num_threads(6);
    while (iterations < MAX_ITER) // iterations start here. in every iteration new rank vector is calculated and stored in r_new vector.
    {
        vector<float> r_new(page_indices.size(), 0);
        float difference = 0;
        int pid;
        int i;
        #pragma omp parallel shared(row_begin, outer_links, col_indices, r_new, difference) private(pid, i)
        {
            // every integer i between row_begin[i] and row_begin[i+1] is the col_indices of ith node. 
            // code block in below for loop calculates the dot product of ith row of P matrix and rank vector.
            // independent and seperate calculations. parallelization is applied here.
            #pragma omp for schedule(dynamic, 20)
            for (i = 0; i < row_begin.size() - 1; i++) 
            {
                r_new.at(i) = 0;
                for (int j = row_begin.at(i); j < row_begin.at(i + 1); j++)
                {
                    r_new.at(i) += (1.0 / outer_links.find(col_indices.at(j))->second) * r.at(col_indices.at(j));
                }
                r_new.at(i) = ALPHA * r_new.at(i) + 1 - ALPHA;
                #pragma omp critical(myregion) // critical because race conditions might happen on writing to difference.
                {
                    difference += abs(r_new.at(i) - r.at(i));
                }
            }
        }
        r = r_new;
        iterations++;
        printf("iteration:\t%d\tdifference:\t%f\n", iterations, difference);
        if (difference < EPSILON) break; // stop when tolerance limit is subceeded.
    }

}

int load_file_to_csr_matrix(char * filename)
{
    vector<pair<string, string>> edges;         // stores the edges in a vector before forming the adjacency matrix in CSR format
    fstream File(filename);
    string line;
    
    // in this function only head nodes are assigned to an id 
    // because we want to store the edges into CSR format without forming
    // another 2D matrix. in this way every head node will be indexed 
    // incerementally.
    while (getline(File, line))                 // file is read and stored into edges while head nodes are assigned to an index in page_indices and page_ids maps.
    {
        string tail_node = line.substr(0, STR_LEN);
        string head_node = line.substr(STR_LEN + 1, STR_LEN);
        int id = page_indices.insert(make_pair(head_node, page_indices.size())).first->second;
        page_ids.insert(make_pair(id, head_node));
        edges.push_back(make_pair(tail_node, head_node));
    }
    time_t here2 = time(NULL);

    
    // in this while loop tail nodes are pushed into the row_begin and 
    // col_indices with their assigned ids. also outer_links is filled
    // with the number of outgoing edges from a node. 
    int old_head_index = -1;                    // used in order to differentiate the prior head node from the current one.
    for (int i = 0; i < edges.size(); i++)
    {
        int head_index = page_indices.find(edges.at(i).second)->second;
        int tail_index = page_indices.insert(make_pair(edges.at(i).first, page_indices.size())).first->second;
        if (tail_index == head_index)
            continue;
        if (head_index != old_head_index)
        {
            old_head_index = head_index;
            row_begin.push_back(col_indices.size());
        }
        page_ids.insert(make_pair(tail_index, edges.at(i).first));
        col_indices.push_back(tail_index);
        outer_links.insert(make_pair(tail_index, 0)).first->second++;
    }

    // rest of the rows are repeated with the last element. 
    // they stand for the number of nodes without any incoming edges to them. 
    row_begin.push_back(col_indices.size());
    for (int i = row_begin.size(); i <= page_indices.size(); i++)
    {
        row_begin.push_back(col_indices.size());
    }
    
    return 0;
}