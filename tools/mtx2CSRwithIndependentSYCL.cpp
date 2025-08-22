#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <ctime>
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <queue>
#include <utility>
#include <vector>
#include <iostream>
#include <vector>
#include <CL/sycl.hpp>


using namespace std;

void printCSRRepresentation(const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers);


void find_neighbors(const std::vector<int>& row_pointers, const std::vector<int>& column_indices, int vertex) {
    int start = row_pointers[vertex];
    int end = row_pointers[vertex + 1];
    std::cout << "Neighbors of vertex " << vertex << ": ";
    for (int i = start; i < end; ++i) {
        std::cout << column_indices[i] << " ";
    }
    std::cout << std::endl;
}




const int INF = 1e9; // Representing infinity

std::vector<std::tuple<int, int, int>> readMTX(const std::string& filename) {
    std::ifstream infile(filename);
    std::vector<std::tuple<int, int, int>> graph;
    
    if (!infile.is_open()) {
        std::cerr << "Failed to open file " << filename << std::endl;
        return graph;
    }
    
    std::string line;
    do {
        std::getline(infile, line);
    } while (line[0] == '%');
    
    int numRows, numCols, numNonZero;
    std::stringstream ss(line);
    ss >> numRows >> numCols >> numNonZero;
    
    int row, col, weight;
    while (infile >> row >> col >> weight) {
        graph.emplace_back(row, col, weight);
    }
    
    return graph;
}

void writeMTX(const std::string& filename, const std::vector<std::tuple<int, int, int>>& graph, int numVertices, bool isGraph) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file " << filename << std::endl;
        return;
    }
    
    outfile << numVertices << " " << numVertices << " " << graph.size() << "\n";
    
    for (const auto& [src, dest, weight] : graph) {
        if (isGraph && weight < 0)
            continue;
        outfile << src << " " << dest << " " << weight << "\n";
    }
}

std::vector<std::tuple<int, int, int>> generateChangedGraph(
    const std::vector<std::tuple<int, int, int>>& originalGraph,
    int numVertices,
    int numChanges,
    int minWeight,
    int maxWeight,
    std::vector<std::tuple<int, int, int>>& changedEdges,
    float deletionPercentage // e.g., 0.2 for 20%
) {
    std::vector<std::tuple<int, int, int>> newGraph = originalGraph;
    std::set<std::pair<int, int>> existingEdges;

    for (const auto& [src, dest, weight] : originalGraph) {
        existingEdges.insert({src, dest});
    }

    std::srand(std::time(nullptr));

    int numDeletions = static_cast<int>(numChanges * deletionPercentage);
    int numOtherActions = numChanges - numDeletions;

    for (int i = 0; i < numChanges; ++i) {
        int action;
        
        if (numDeletions > 0) {
            action = 2;
            numDeletions--;
        } else if (numOtherActions > 0) {
            action = std::rand() % 2; // 0 or 1
            numOtherActions--;
        }

        if (action == 0 && !newGraph.empty()) {
            // Change Weight
            int index = std::rand() % newGraph.size();
            int newWeight = minWeight + std::rand() % (maxWeight - minWeight + 1);
            std::get<2>(newGraph[index]) = newWeight;
            changedEdges.push_back(newGraph[index]);
        } else if (action == 1) {
            // Add Edge
            int src, dest;
            do {
                src = 1 + std::rand() % numVertices;
                dest = 1 + std::rand() % numVertices;
            } while (src == dest || existingEdges.find({src, dest}) != existingEdges.end());

            int newWeight = minWeight + std::rand() % (maxWeight - minWeight + 1);

            newGraph.emplace_back(src, dest, newWeight);
            changedEdges.emplace_back(src, dest, newWeight);
            existingEdges.insert({src, dest});
        } else {
            // Delete Edge (by setting the weight to the negative of the current weight)
            if (!newGraph.empty()) {
                int index = std::rand() % newGraph.size();
                int curWeight = std::get<2>(newGraph[index]);
                std::get<2>(newGraph[index]) = -curWeight;
                changedEdges.push_back(newGraph[index]);
            }
        }
    }

    std::sort(newGraph.begin(), newGraph.end());
    std::sort(changedEdges.begin(), changedEdges.end());

    return newGraph;
}

void sortAndSaveMTX(const std::string& input_filename, const std::string& output_filename) {
    std::ifstream infile(input_filename);

    if (!infile.is_open()) {
        std::cerr << "Failed to open the input file." << std::endl;
        return;
    }

    std::string line;
    int numRows, numCols, numNonZero;

    // Skip comments
    do {
        std::getline(infile, line);
    } while (line[0] == '%');

    std::stringstream ss(line);
    ss >> numRows >> numCols >> numNonZero;

    std::vector<std::tuple<int, int, int>> edges;  // source, destination, weight
    int row, col, weight;
    int prev_row = -1;
    bool is_sorted = true;

    while (infile >> row >> col >> weight) {
        if (row < prev_row) {
            is_sorted = false;
        }
        prev_row = row;
        edges.emplace_back(row, col, weight);
    }

    infile.close();

    if (!is_sorted) {
        std::sort(edges.begin(), edges.end());
    }

    // Save to a new MTX file
    std::ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the output file." << std::endl;
        return;
    }

    outfile << numRows << " " << numCols << " " << numNonZero << std::endl;

    for (const auto& [r, c, w] : edges) {
        outfile << r << " " << c << " " << w << std::endl;
    }

    outfile.close();
}

#include <iostream>
#include <vector>
#include <utility>  // for std::pair

#include <iostream>
#include <vector>
#include <utility>  // for std::pair
#include <algorithm> // for std::remove

std::vector<std::vector<std::pair<int, int>>> CSRToAdjacencyList( std::vector<int>& values,
     std::vector<int>& column_indices,
     std::vector<int>& row_pointers) {

    std::vector<std::vector<std::pair<int, int>>> adjList(row_pointers.size() - 1);

    for (int i = 0; i < row_pointers.size() - 1; ++i) {
        for (int j = row_pointers[i]; j < row_pointers[i + 1]; ++j) {
            adjList[i].push_back({column_indices[j], values[j]});
        }
    }

    return adjList;
}

void adjacencyListToCSR( std::vector<std::vector<std::pair<int, int>>>& adjList,
    std::vector<int>& values,
    std::vector<int>& column_indices,
    std::vector<int>& row_pointers) {
    values.clear();
    column_indices.clear();
    row_pointers.clear();

    row_pointers.push_back(0);

    for ( auto& neighbors : adjList) {
        for ( auto& [neighbor, weight] : neighbors) {
            values.push_back(weight);
            column_indices.push_back(neighbor);
        }
        row_pointers.push_back(column_indices.size());
    }
}

void removeEdgeAndUpdateCSR(int u, int v,
                            std::vector<int>& values,
                            std::vector<int>& column_indices,
                            std::vector<int>& row_pointers) {
    // Convert CSR to adjacency list
    auto adjList = CSRToAdjacencyList(values, column_indices, row_pointers);

    // Remove edge (u, v)
    adjList[u].erase(std::remove_if(adjList[u].begin(), adjList[u].end(),
                                    [v](const std::pair<int, int>& p) { return p.first == v; }),
                     adjList[u].end());

    // Convert back to CSR
    adjacencyListToCSR(adjList, values, column_indices, row_pointers);
}

void addEdgeAndUpdateCSR(int u, int v, int w,
                         std::vector<int>& values,
                         std::vector<int>& column_indices,
                         std::vector<int>& row_pointers) {
    // Convert CSR to adjacency list
    auto adjList = CSRToAdjacencyList(values, column_indices, row_pointers);

    // Add edge (u, v, w)
    adjList[u].push_back({v, w});

    // Convert back to CSR
    adjacencyListToCSR(adjList, values, column_indices, row_pointers);
}

std::vector<std::vector<int>> CSRToInDegreeList(const std::vector<int>& values,
                                                const std::vector<int>& column_indices,
                                                const std::vector<int>& row_pointers) {
    int n = row_pointers.size() - 1; // Number of vertices
    std::vector<std::vector<int>> inDegreeList(n); // Initialize in-degree list

    for (int u = 0; u < n; ++u) { // Loop through each vertex
        int start = row_pointers[u] ;  // 1-indexed to 0-indexed
        int end = row_pointers[u + 1] ;  // 1-indexed to 0-indexed
        for (int i = start; i < end; ++i) {
            int v = column_indices[i];  // Retrieve adjacent vertex
            inDegreeList[v].push_back(u );  // Fill in the in-degree list, convert to 1-indexed
        }
    }

    return inDegreeList;
}
#include <vector>
#include <iostream>
#include <limits>

int getWeightFromCSR(
    const std::vector<int>& values, 
    const std::vector<int>& column_indices, 
    const std::vector<int>& row_pointers, 
    int u, 
    int v) 
{
    for (int i = row_pointers[u]; i < row_pointers[u + 1]; ++i) {
        if (column_indices[i] == v) {
            //std::cout<< "Found "<< u << " to "<< v << "with values "<< values[i]<< std::endl;
            return values[i];
        }
    }
    std::cerr << "Edge (" << u << ", " << v << ") not found." << std::endl;
    return std::numeric_limits<int>::infinity();
}


#include <tuple>

void markSubtreeAffected(const std::vector<int>& sssp_values, 
                         const std::vector<int>& sssp_column_indices, 
                         const std::vector<int>& sssp_row_pointers, 
                         std::vector<int>& dist, 
                         std::vector<bool>& isAffectedForDeletion, 
                         std::queue<int>& affectedNodesForDeletion, 
                         int node) {


    
    dist[node] = INT_MAX; // Invalidate the shortest distance
    isAffectedForDeletion[node] = true;
    affectedNodesForDeletion.push(node);
  
    // Get the start and end pointers for the row in CSR representation
    int start = sssp_row_pointers[node]; // Already 1-indexed
    int end = sssp_row_pointers[node + 1]; // Already 1-indexed

    // Traverse the CSR to find the children of the current node
    for (int i = start; i < end; ++i) {
        int child = sssp_column_indices[i]; // Already 1-indexed
        //std::cout<< child << " "<<std::endl;
        // If this child node is not already marked as affected, call the function recursively
        if (!isAffectedForDeletion[child]) {
            markSubtreeAffected(sssp_values, sssp_column_indices, sssp_row_pointers, dist, isAffectedForDeletion, affectedNodesForDeletion, child);
        }
    }
}

struct CSR {
    vector<int> values;
    vector<int> row_ptr;
    vector<int> col_idx;
};

vector<CSR> singleColumnCSR(const CSR& csr, int n_columns) {
    vector<CSR> singleColumnCSRs(n_columns);

    for (int i = 0; i < n_columns; ++i) {
        vector<int> values_i;
        vector<int> row_ptr_i = {0};
        vector<int> col_idx_i;

        for (int j = 0; j < csr.row_ptr.size() - 1; ++j) {
            int row_start = csr.row_ptr[j];
            int row_end = csr.row_ptr[j + 1];

            bool found = false;
            for (int k = row_start; k < row_end; ++k) {
                if (csr.col_idx[k] == i) {
                    values_i.push_back(csr.values[k]);
                    col_idx_i.push_back(0);
                    found = true;
                    break;
                }
            }

            if (found) {
                row_ptr_i.push_back(values_i.size());
            } else {
                row_ptr_i.push_back(row_ptr_i.back());
            }
        }

        singleColumnCSRs[i] = {values_i, row_ptr_i, col_idx_i};
    }

    return singleColumnCSRs;
}

CSR combineSingleColumnCSR(const vector<CSR>& singleColumnCSRs) {
    CSR original;
    int n_rows = singleColumnCSRs[0].row_ptr.size() - 1;

    original.row_ptr.push_back(0);

    for (int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < singleColumnCSRs.size(); ++col) {
            const CSR& singleColumn = singleColumnCSRs[col];

            int row_start = singleColumn.row_ptr[row];
            int row_end = singleColumn.row_ptr[row + 1];

            for (int k = row_start; k < row_end; ++k) {
                original.values.push_back(singleColumn.values[k]);
                original.col_idx.push_back(col);
            }
        }
        original.row_ptr.push_back(original.values.size());
    }

    return original;
}

void updateShortestPath( std::vector<int>& new_graph_values,  std::vector<int>& new_graph_column_indices,  std::vector<int>& new_graph_row_pointers,
                         std::vector<int>& sssp_values,  std::vector<int>& sssp_column_indices,  std::vector<int>& sssp_row_pointers,
                         std::vector<int>& ce_graph_values,  std::vector<int>& ce_graph_column_indices,  std::vector<int>& ce_graph_row_pointers,
                        std::vector<int>& dist, std::vector<int>& parent, std::vector<std::tuple<int, int, int>> changedEdges , std::vector<std::vector<int>> predecessor) {

    // std::cout << "Distance before" <<std::endl;
    // for (int i = 0; i < dist.size(); i++) {
    //     std::cout <<dist[i]<< " ";
    // }
    // std::cout<<std::endl;

    // std::cout << "Parent before " <<std::endl;
    // for (int i = 0; i < parent.size(); i++) {
    //     std::cout <<parent[i]<< " ";
    // }
    // std::cout<<std::endl;
  
    // Convert changedEdges to CSR for insertion and deletion separately
    std::vector<int> insert_values, delete_values;
    std::vector<int> insert_column_indices, delete_column_indices;
    std::vector<int> insert_row_pointers, delete_row_pointers;
    insert_row_pointers.push_back(0);
    delete_row_pointers.push_back(0);

    int insert_nnz = 0, delete_nnz = 0;

    for (int u = 0; u < new_graph_row_pointers.size() - 1; ++u) {
        for (const auto& [src, dest, weight] : changedEdges) {
            if (src - 1 == u) {
                if (weight >= 0) {
                    insert_values.push_back(weight);
                    insert_column_indices.push_back(dest - 1);
                    insert_nnz++;
                } else {
                    delete_values.push_back(-weight);
                    delete_column_indices.push_back(dest - 1);
                    delete_nnz++;
                }
            }
        }
        insert_row_pointers.push_back(insert_nnz);
        delete_row_pointers.push_back(delete_nnz);
    }

    //printCSRRepresentation(insert_values, insert_column_indices, insert_row_pointers);
    //printCSRRepresentation(delete_values, delete_column_indices, delete_row_pointers);


    // Original logic of updateShortestPath adapted to use changed_* vectors
    std::queue<int> affectedNodes;
    std::vector<bool> isAffected(new_graph_row_pointers.size() - 1, false);

    //printCSRRepresentation(sssp_values, sssp_column_indices, sssp_row_pointers);

    // Convert to a set of independent CSRs
    CSR insert = {
        insert_values,
        insert_column_indices,
        insert_row_pointers
    };
    vector<CSR> singleColumnMatricesInsert = singleColumnCSR(insert, insert_row_pointers.size() - 1 );

    for (int i = 0; i < singleColumnMatricesInsert.size(); ++i) {
        auto insert_val = singleColumnMatricesInsert[i].values;
        auto insert_col = singleColumnMatricesInsert[i].col_idx; 
        auto insert_row_pt = singleColumnMatricesInsert[i].row_ptr;


        for (int u = 0; u < new_graph_row_pointers.size() - 1; ++u) {
        int start = insert_row_pt[u];
        int end = insert_row_pt[u + 1];

            for (int i = start; i < end; ++i) {
                int v = insert_col[i];
                int alt = dist[u] + insert_val[i];
                int w = new_graph_values[i];
                if (alt < dist[v]) {

                    //std::cout<< u + 1 << " to "<< v + 1 << " becomes " << alt << " from "<< dist[v]<< std::endl; 
                    //std:: cout<< "Old parent" << parent[v] << " of " << v << " and "<< "New parent"<< u<<std::endl;
                    removeEdgeAndUpdateCSR(parent[v], v, sssp_values, sssp_column_indices, sssp_row_pointers);
                    addEdgeAndUpdateCSR(u, v, dist[v] - dist[u], sssp_values, sssp_column_indices, sssp_row_pointers);
                    dist[v] = alt;
                    parent[v] = u;
                    isAffected[v] = true;
                    affectedNodes.push(v);
                }
            }
        }
        //printCSRRepresentation(sssp_values, sssp_column_indices, sssp_row_pointers);

        // Propagate changes for insertion
        while (!affectedNodes.empty()) {
            int u = affectedNodes.front();
            affectedNodes.pop();
            isAffected[u] = false;

            int start = new_graph_row_pointers[u];
            int end = new_graph_row_pointers[u + 1];

            for (int i = start; i < end; ++i) {
                int v = new_graph_column_indices[i];
                int w = new_graph_values[i];
                int alt = dist[u] + new_graph_values[i];
                if (alt < dist[v]) {
                    //std::cout<< u + 1 << " to "<< v + 1 << " becomes " << alt << " from "<< dist[v]<< std::endl;

                    removeEdgeAndUpdateCSR(parent[v], v, sssp_values, sssp_column_indices, sssp_row_pointers);
                    addEdgeAndUpdateCSR(u, v, w, sssp_values, sssp_column_indices, sssp_row_pointers);
                    dist[v] = alt;
                    parent[v] = u;
                    if (!isAffected[v]) {
                        isAffected[v] = true;
                        affectedNodes.push(v);
                    }
                }
            }
        }

    }

    

    CSR deletion = {
        delete_values,
        delete_column_indices,
        delete_row_pointers
    };
    vector<CSR> singleColumnMatricesDeletion = singleColumnCSR(deletion, delete_row_pointers.size() - 1 );

    for (int i = 0; i < singleColumnMatricesDeletion.size(); ++i) {
        auto delete_val = singleColumnMatricesDeletion[i].values;
        auto delete_col = singleColumnMatricesDeletion[i].col_idx; 
        auto delete_row_pt = singleColumnMatricesDeletion[i].row_ptr;

        // Handle deletions
        std::queue<int> affectedNodesForDeletion;
        std::vector<bool> isAffectedForDeletion(new_graph_row_pointers.size() - 1, false);

        for (int u = 0; u < new_graph_row_pointers.size() - 1; ++u) {
            int start = delete_row_pt[u];
            int end = delete_row_pt[u + 1];


            for (int i = start; i < end; ++i) {
                int v = delete_col[i];
                if (parent[v] == u) { // if this deleted edge was part of the shortest path

                    // std::cout<< u << " to "<< v << std::endl;
                    markSubtreeAffected(sssp_values, sssp_column_indices, sssp_row_pointers, dist, isAffectedForDeletion, affectedNodesForDeletion, v);
                    // find new parent if exist

                    int newDistance = INT_MAX;
                    int newParentIndex = -1;

                    for ( int i = 0; i < predecessor[v].size(); i++)
                    {
                        if(dist[predecessor[v][i]] + getWeightFromCSR(new_graph_values, new_graph_column_indices, new_graph_row_pointers, predecessor[v][i], v) < newDistance )
                        {
                            newDistance = dist[predecessor[v][i]] + getWeightFromCSR(new_graph_values, new_graph_column_indices, new_graph_row_pointers, predecessor[v][i], v);
                            newParentIndex = predecessor[v][i]; 
                            //std::cout<< "New parent found"<< newParentIndex << " with distance " << dist[predecessor[v][i]] << " + "<< getWeightFromCSR(new_graph_values, new_graph_column_indices, new_graph_row_pointers, predecessor[v][i], v) <<std::endl;
                        }
                    }
                    int oldParent = parent[v];
                    if (newParentIndex == -1)
                    {
                        parent[v] = -1; 
                        dist[v] = INT_MAX; 
                    }
                    else
                    {
                        dist[v] = newDistance;
                        removeEdgeAndUpdateCSR(oldParent, v, sssp_values, sssp_column_indices, sssp_row_pointers);
                        addEdgeAndUpdateCSR(newParentIndex, v, newDistance - dist[newParentIndex] , sssp_values, sssp_column_indices, sssp_row_pointers);
                    }
                    parent[v] = newParentIndex;

                    // update sssp
                }
            }
        }
        while (!affectedNodesForDeletion.empty()) {
        int u = affectedNodesForDeletion.front();
        affectedNodesForDeletion.pop();
        isAffectedForDeletion[u] = false;

        int start = new_graph_row_pointers[u];
        int end = new_graph_row_pointers[u + 1];

        for (int i = start; i < end; ++i) {
            int v = new_graph_column_indices[i];
            int w = new_graph_values[i];
            int alt = dist[u] + new_graph_values[i];
            if (dist[v] == INT_MAX) {
                //std::cout<< u + 1 << " to "<< v + 1 << " becomes " << alt << " from "<< dist[v]<< std::endl;

                int newDistance = INT_MAX;
                int newParentIndex = -1;

                for ( int i = 0; i < predecessor[v].size(); i++)
                {
                    if(dist[predecessor[v][i]] + getWeightFromCSR(new_graph_values, new_graph_column_indices, new_graph_row_pointers, predecessor[v][i], v) < newDistance )
                    {
                        newDistance = dist[predecessor[v][i]] + getWeightFromCSR(new_graph_values, new_graph_column_indices, new_graph_row_pointers, predecessor[v][i], v);
                        newParentIndex = predecessor[v][i]; 
                        //std::cout<< "New parent found"<< newParentIndex << " with distance " << dist[predecessor[v][i]] << " + "<< getWeightFromCSR(new_graph_values, new_graph_column_indices, new_graph_row_pointers, predecessor[v][i], v) <<std::endl;
                    }
                }
                if ( v + 1 == 1)
                    continue;

                int oldParent = parent[v];
                if (newParentIndex == -1)
                {
                    parent[v] = -1; 
                    dist[v] = INT_MAX; 
                }
                else
                {
                    dist[v] = newDistance;
                    removeEdgeAndUpdateCSR(oldParent, v, sssp_values, sssp_column_indices, sssp_row_pointers);
                    addEdgeAndUpdateCSR(newParentIndex, v, newDistance - dist[newParentIndex] , sssp_values, sssp_column_indices, sssp_row_pointers);
                }
                parent[v] = newParentIndex;
                if (!isAffected[v]) {
                    isAffected[v] = true;
                    affectedNodes.push(v);
                }
                //Testing needed
            }
        }
    }

    }

    

     
    


    // std::cout << "Distance after" <<std::endl;
    // for (int i = 0; i < dist.size(); i++) {
    //     std::cout <<dist[i]<< " ";
    // }
    // std::cout<<std::endl;

    // std::cout << "Parent after " <<std::endl;
    // for (int i = 0; i < parent.size(); i++) {
    //     std::cout <<parent[i]<< " ";
    // }
    // std::cout<<std::endl;

}




#include <vector>
#include <queue>
#include <utility>
#include <limits>

void dijkstra(const std::vector<int>& values, const std::vector<int>& column_indices, 
              const std::vector<int>& row_pointers, int src,
              std::vector<int>& dist, std::vector<int>& parent)
{
    // Initialize the distance vector
    int n = row_pointers.size() - 1;
    dist.resize(n, std::numeric_limits<int>::max());
    dist[src] = 0;

    // Initialize the parent vector
    parent.resize(n, -1);
    parent[src] = src;

    // Priority queue to store {distance, vertex} pairs
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
    pq.push({0, src});

    while (!pq.empty()) {
        int u = pq.top().second;
        int uDist = pq.top().first;
        pq.pop();

        // Ignore if distance in the queue is outdated
        if (uDist > dist[u]) continue;

        int start = row_pointers[u];
        int end = row_pointers[u + 1];

        for (int i = start; i < end; ++i) {
            int v = column_indices[i];
            int weight = values[i];
            
            // Relaxation step
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
}  
int numRows, numCols, numNonZero;
bool readMTXToCSR(const std::string& filename, std::vector<int>& values, std::vector<int>& column_indices, std::vector<int>& row_pointers) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return false;
    }

    std::string line;
    int numRows, numCols, numNonZero;

    // Skip comments
    do {
        std::getline(file, line);
    } while (line[0] == '%');

    std::stringstream ss(line);
    ss >> numRows >> numCols >> numNonZero;

    values.resize(numNonZero);
    column_indices.resize(numNonZero);
    row_pointers.resize(numRows + 1, 0);

    int row, col, val;
    int nnz = 0;
    int current_row = 0;

    for (int i = 0; i < numNonZero; ++i) {
        file >> row >> col >> val;

        // Convert to 0-based indexing
        row -= 1;
        col -= 1;

        while (row > current_row) {
            row_pointers[current_row + 1] = nnz;
            current_row++;
        }

        values[nnz] = val;
        column_indices[nnz] = col;
        nnz++;
    }

    // Add the last row_pointer
    row_pointers[current_row + 1] = nnz;

    // Close the file
    file.close();

    return true;
}
// Function to print CSR representation
void printCSRRepresentation(const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers) {
    std::cout << "CSR representation of the Graph:\n";
    std::cout << "Values: ";
    for (int val : values) {
        std::cout << val << " ";
    }
    std::cout << "\nColumn Indices: ";
    for (int col : column_indices) {
        std::cout << col << " ";
    }
    std::cout << "\nRow Pointers: ";
    for (int row_ptr : row_pointers) {
        std::cout << row_ptr << " ";
    }
    std::cout << std::endl;
}

void saveSSSPTreeToFile(const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers, const std::vector<int>& parent) {
    std::ofstream outfile("SSSP_Tree.mtx");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the output file." << std::endl;
        return;
    }

    outfile << parent.size() << " " << parent.size() << " " << parent.size() - 1 << std::endl;

    // index starting from 1 as the source is a parent of the source case
    for (int i = 1; i < parent.size(); i++) {
        int val = -1;
        int start = row_pointers[parent[i]];
        int end = row_pointers[parent[i] + 1];
        for (; start < end; start++) {
            if (column_indices[start] == i) {
                val = values[start];
            }
        }
        outfile << parent[i] + 1 << " " << i + 1 << " " << val << std::endl;
    }

    outfile.close();
}
void saveSSSPTreeToFile(const std::string& fileName, const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers, const std::vector<int>& parent) {
    std::ofstream outfile(fileName);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open the output file." << std::endl;
        return;
    }

    outfile << parent.size() << " " << parent.size() << " " << parent.size() - 1 << std::endl;

    // index starting from 1 as the source is a parent of the source case
    for (int i = 1; i < parent.size(); i++) {
        int val = -1;
        int start = row_pointers[parent[i]];
        int end = row_pointers[parent[i] + 1];
        for (; start < end; start++) {
            if (column_indices[start] == i) {
                val = values[start];
            }
        }
        outfile << parent[i] + 1 << " " << i + 1 << " " << val << std::endl;
    }

    outfile.close();
}

int main() {

//Input Graph to CSR

    sortAndSaveMTX("graph.mtx", "sorted_graph.mtx");

    std::vector<int> values;
    std::vector<int> column_indices;
    std::vector<int> row_pointers;

    readMTXToCSR("sorted_graph.mtx", values, column_indices, row_pointers);
    //printCSRRepresentation(values, column_indices, row_pointers);

   

    


//Find SSSP tree and store in mtx file


    std::vector<int> parent(row_pointers.size() - 1, -1); // takes child and returns it's parent
    std::vector<int> dist(row_pointers.size() - 1, INT_MAX);
    
    // Run Dijkstra's algorithm from source vertex 0
    dijkstra(values, column_indices, row_pointers, 0, dist, parent);

    saveSSSPTreeToFile("SSSP_Tree.mtx", values, column_indices, row_pointers, parent);

    std::vector<int> sssp_values;
    std::vector<int> sssp_column_indices;
    std::vector<int> sssp_row_pointers;
    sortAndSaveMTX("SSSP_Tree.mtx", "sorted_SSSP_Tree.mtx");
    readMTXToCSR("sorted_SSSP_Tree.mtx", sssp_values, sssp_column_indices, sssp_row_pointers);
    //printCSRRepresentation(sssp_values, sssp_column_indices, sssp_row_pointers);

// Changed edges
    auto originalGraph = readMTX("sorted_graph.mtx");

    int numVertices = row_pointers.size() - 1;  // Should be determined from the MTX file or another source
    int numChanges = 3;
    int minWeight = 1;
    int maxWeight = 10;

    std::vector<std::tuple<int, int, int>> changedEdges;
    float deletePercentage = 0.0f;
    auto newGraph = generateChangedGraph(originalGraph, numVertices, numChanges, minWeight, maxWeight, changedEdges, deletePercentage);
    // writeMTX by default sort by row for easy reading
    writeMTX("new_graph.mtx", newGraph, numVertices, true); 
    writeMTX("changed_edges.mtx", changedEdges, numVertices, false);

    std::vector<int> new_graph_values;
    std::vector<int> new_graph_column_indices;
    std::vector<int> new_graph_row_pointers;
    readMTXToCSR("new_graph.mtx", new_graph_values, new_graph_column_indices, new_graph_row_pointers);
    //printCSRRepresentation(new_graph_values, new_graph_column_indices, new_graph_row_pointers);

    std::vector<int> ce_graph_values;
    std::vector<int> ce_graph_column_indices;
    std::vector<int> ce_graph_row_pointers;
    readMTXToCSR("changed_edges.mtx", ce_graph_values, ce_graph_column_indices, ce_graph_row_pointers);
    //printCSRRepresentation(ce_graph_values, ce_graph_column_indices, ce_graph_row_pointers);

// Update shortest path code
    

    // Now call the updateShortestPath function
    auto inDegreeList = CSRToInDegreeList(new_graph_values, new_graph_column_indices, new_graph_row_pointers);
    updateShortestPath(new_graph_values, new_graph_column_indices, new_graph_row_pointers, sssp_values, sssp_column_indices, sssp_row_pointers, ce_graph_values, ce_graph_column_indices, ce_graph_row_pointers, dist, parent, {
        std::make_tuple(2, 3, 9),
        std::make_tuple(3, 1, 6),
        std::make_tuple(1, 3, -5)
    }, inDegreeList);

    std::vector<int> new_values, new_column_indices, new_row_pointers;

    new_row_pointers.push_back(0);
    int nnz = 0;
    for (int u = 0; u < numRows; ++u) {
        if (parent[u] != -1) {
            new_values.push_back(dist[u]);  // storing the shortest distance as the value
            new_column_indices.push_back(parent[u]);  // storing the parent as the column index
            nnz++;
        }
        new_row_pointers.push_back(nnz);
    }

    // Print new CSR representation
    //printCSRRepresentation(new_values, new_column_indices, new_row_pointers);

    return 0;
}

//clang++ -std=c++17 mtx2CSR.cpp  && ./a.out