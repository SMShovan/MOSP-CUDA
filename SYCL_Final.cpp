#include <bits/stdc++.h>
#include <CL/sycl.hpp>
#include <fstream>
#include <ctime>
using namespace cl::sycl;
using namespace std;
const int INF = 1e9;
using Graph = std::vector<std::vector<std::pair<int, int>>>;

// Timing structure to store all timing data
struct TimingData {
    double updateShortestPath_time;
    double constructGraph_time;
    double updateWeights_time;
    double parallelBellmanFord_time;
    double total_time;
};

// Global timing data
TimingData globalTiming;

struct CSR {
    vector<int> values;
    vector<int> row_ptr;
    vector<int> col_idx;
};
struct Edge {
    int u;
    int v;
    float weight;
    int whichTree;
    Edge(int u, int v, float weight, int whichTree) : u(u), v(v), weight(weight), whichTree(whichTree) {}
};

class Tree {
public:
    virtual const std::vector<int>& getParents() const = 0;
    virtual ~Tree() = default;
};

class Tree1 : public Tree {
    std::vector<int> parents;
public:
    Tree1(const std::vector<int>& p) : parents(p) {}
    const std::vector<int>& getParents() const override {
        return parents;
    }
};

class Tree2 : public Tree {
    std::vector<int> parents;
public:
    Tree2(const std::vector<int>& p) : parents(p) {}
    const std::vector<int>& getParents() const override {
        return parents;
    }
};

class Tree3 : public Tree {
    std::vector<int> parents;
public:
    Tree3(const std::vector<int>& p) : parents(p) {}
    const std::vector<int>& getParents() const override {
        return parents;
    }
};
// Function to add an edge to the graph
void addEdge(std::vector<Edge>& edges, int u, int v, double pref, int index) {
    edges.push_back({u, v, static_cast<float>(pref), index});
}

// Function to construct the graph
std::vector<Edge> constructGraph(const std::vector<Tree*>& trees, int k) {
    std::vector<Edge> edges;

    for (size_t index = 0; index < trees.size(); ++index) {
        const auto& tree = trees[index];
        const std::vector<int>& parents = tree->getParents();

        for (int i = 1; i < parents.size(); ++i) {
            if (parents[i] != 0) {
                addEdge(edges, parents[i], i, k, index);
            }
        }
    }

    return edges;
}



void updateWeights(std::vector<Edge>& edges, std::vector<int>& Pref, int k) {
    
    clock_t start = clock();

    cl::sycl::queue q(cl::sycl::default_selector{});
    const size_t size = edges.size();

    // Correctly initialized buffers
    cl::sycl::buffer<Edge, 1> edges_buf(edges.data(), cl::sycl::range<1>(size));
    cl::sycl::buffer<int, 1> weights_buf((cl::sycl::range<1>(size)));
    cl::sycl::buffer<int, 1> index_buf((cl::sycl::range<1>(size))); 
    cl::sycl::buffer<int, 1> pref_buf(Pref.data(), cl::sycl::range<1>(Pref.size()));


    // Step 1: Parallel copy of weights from edges to weights buffer
    q.submit([&](cl::sycl::handler& cgh) {
        auto e_acc = edges_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto w_acc = weights_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
        auto i_acc = index_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for<class copy_weights_kernel>(cl::sycl::range<1>(size), [=](cl::sycl::id<1> idx) {
            w_acc[idx] = e_acc[idx].weight;
            i_acc[idx] = e_acc[idx].whichTree;
        });
    });

    // Step 2: Perform atomic subtraction on weights
    q.submit([&](cl::sycl::handler& cgh) {
        auto acc = weights_buf.get_access<cl::sycl::access::mode::atomic>(cgh);
        auto i_acc = index_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
        auto pref_acc = pref_buf.get_access<cl::sycl::access::mode::read>(cgh);
        cgh.parallel_for<class sub_weights_kernel>(cl::sycl::range<1>(size), [=](cl::sycl::id<1> idx) {
            acc[idx].fetch_sub(1/pref_acc[i_acc[idx]]);
        });
    });

    // Step 3: Parallel update of original edges with new weights
    q.submit([&](cl::sycl::handler& cgh) {
        auto e_acc = edges_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto w_acc = weights_buf.get_access<cl::sycl::access::mode::read>(cgh);
        cgh.parallel_for<class update_edges_kernel>(cl::sycl::range<1>(size), [=](cl::sycl::id<1> idx) {
            e_acc[idx].weight = w_acc[idx];
        });
    });

    clock_t end = clock();
    globalTiming.updateWeights_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    q.wait();
}

void parallelBellmanFord(std::vector<Edge>& edges, int numVertices, int source, std::vector<int>& parents) {
    
    clock_t start = clock();

    // Initialize distances to infinity, except for the source vertex
    std::vector<float> distances(numVertices, std::numeric_limits<float>::max());
    distances[source] = 0.0f;

    // Initialize parent array with -1, indicating no parent initially
    parents.assign(numVertices, -1);
    parents[source] = source; // The source is its own parent

    // Create buffers for edges, distances, and parents
    cl::sycl::buffer<Edge, 1> edges_buf(edges.data(), cl::sycl::range<1>(edges.size()));
    cl::sycl::buffer<float, 1> distances_buf(distances.data(), cl::sycl::range<1>(distances.size()));
    cl::sycl::buffer<int, 1> parents_buf(parents.data(), cl::sycl::range<1>(parents.size()));

    // SYCL queue for executing kernels
    cl::sycl::queue q(cl::sycl::default_selector{});

    // Main loop of the Bellman-Ford algorithm, executed V-1 times
    for (int i = 0; i < numVertices - 1; ++i) {
        // Submit a task for parallel edge relaxation
        q.submit([&](cl::sycl::handler& cgh) {
            auto edges_acc = edges_buf.get_access<cl::sycl::access::mode::read>(cgh);
            auto dist_acc = distances_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto parents_acc = parents_buf.get_access<cl::sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class relax_edges>(cl::sycl::range<1>(edges.size()), [=](cl::sycl::id<1> idx) {
                int u = edges_acc[idx].u;
                int v = edges_acc[idx].v;
                float weight = edges_acc[idx].weight;

                // Perform relaxation and update parent if a shorter path is found
                if (dist_acc[u] != std::numeric_limits<float>::max() && dist_acc[u] + weight < dist_acc[v]) {
                    dist_acc[v] = dist_acc[u] + weight; // Relax the edge
                    parents_acc[v] = u; // Update parent
                }
            });
        });

        // Wait for the queue to finish processing
        q.wait();
    }
    clock_t end = clock();
    globalTiming.parallelBellmanFord_time = ((double)(end - start)) / CLOCKS_PER_SEC;
}



//Done
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
bool readMTXToTransposeCSR(const std::string& filename, 
                           std::vector<int>& values, 
                           std::vector<int>& row_indices, 
                           std::vector<int>& col_pointers, int flag = 0) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return false;
    }

    std::string line;
    int numRows, numCols, numNonZero;

    do {
        std::getline(file, line);
    } while (line[0] == '%');

    std::stringstream ss(line);
    ss >> numRows >> numCols >> numNonZero;

    std::multimap<int, std::pair<int, int>> transposedEntries;

    int row, col, val;
    for (int i = 0; i < numNonZero; ++i) {
        file >> row >> col >> val;
        row--; // Convert to 0-based indexing
        col--;
        if ( flag == 0)
            transposedEntries.insert({col, {row, val}});
        else if (flag == 1 && val >= 0)
            transposedEntries.insert({col, {row, val}});
        else if (flag == 2 && val < 0)
            transposedEntries.insert({col, {row, val}});
    }

    file.close();

    values.clear();
    row_indices.clear();
    col_pointers.clear();
    col_pointers.clear();

    int current_col = -1;
    for(const auto& entry: transposedEntries) {
        int col = entry.first;
        int row = entry.second.first;
        int value = entry.second.second;

        if(col > current_col) {
            for(int i = 0; i < (col - current_col); ++i) 
                col_pointers.push_back(values.size());
            current_col = col;
        }

        values.push_back(value);
        row_indices.push_back(row);
    }

    col_pointers.push_back(values.size()); // The last element of col_pointers
    return true;
}
std::vector<std::tuple<int, int, int>> generateChangedGraph(
    const std::vector<std::tuple<int, int, int>>& originalGraph,
    int numVertices,
    int numChanges,
    int minWeight,
    int maxWeight,
    std::vector<std::tuple<int, int, int>>& changedEdges,
    float deletionPercentage) {
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
//Done



// Need to check
bool neighborConvertToTransposeCSR(const std::vector<std::vector<int>>& edges,
                                   std::vector<int>& values,
                                   std::vector<int>& row_indices,
                                   std::vector<int>& col_pointers) {
    std::multimap<int, std::pair<int, int>> transposedEntries;
    for (const auto& edge : edges) {
        int row = edge[0]; // source is row
        int col = edge[1]; // destination is col
        int val = edge[2]; // weight is val
        transposedEntries.insert({col, {row, val}});
    }

    int numCols = 0;
    for (const auto& entry: transposedEntries) {
        numCols = std::max(numCols, entry.first + 1);
    }

    values.clear();
    row_indices.clear();
    col_pointers.clear();

    // Initialize col_pointers with zeros
    col_pointers.resize(numCols + 1, 0);

    int current_col = -1;
    for (const auto& entry: transposedEntries) {
        int col = entry.first;
        int row = entry.second.first;
        int value = entry.second.second;

        if (col > current_col) {
            for (int i = current_col + 1; i <= col; ++i)
                col_pointers[i] = values.size();
            current_col = col;
        }

        values.push_back(value);
        row_indices.push_back(row);
    }
    col_pointers[numCols] = values.size(); // The last element of col_pointers
    return true;
}
std::vector<std::vector<int>> find_outgoing_connections(
    const std::vector<int> &values, 
    const std::vector<int> &row_ptr, 
    const std::vector<int> &col_idx, 
    const std::unordered_set<int> &vertices) {
    std::vector<std::vector<int>> outgoing_connections;
    for (const int vertex : vertices) {
        int start = row_ptr[vertex];
        int end = row_ptr[vertex + 1];

        for (int i = start; i < end; ++i) {
            int adjacent_vertex = col_idx[i];
            int weight = values[i];
            outgoing_connections.push_back({vertex, adjacent_vertex, weight});
        }
    }
    return outgoing_connections;
}

// Need to remove recursion
void markSubtreeAffected(const std::vector<int>& outDegreeValues, 
                         const std::vector<int>& outDegreeIndices, 
                         const std::vector<int>& outDegreePointers, 
                         std::vector<int>& dist, 
                         std::vector<bool>& isAffectedForDeletion, 
                         std::queue<int>& affectedNodesForDeletion, 
                         int node) {

    dist[node] = INT_MAX; // Invalidate the shortest distance
    isAffectedForDeletion[node] = true;
    affectedNodesForDeletion.push(node);
  
    // Get the start and end pointers for the row in CSR representation
    int start = outDegreePointers[node]; // Already 1-indexed
    int end = outDegreePointers[node + 1]; // Already 1-indexed

    // Traverse the CSR to find the children of the current node
    for (int i = start; i < end; ++i) {
        int child = outDegreeIndices[i]; // Already 1-indexed
        //std::cout<< child << " "<<std::endl;
        // If this child node is not already marked as affected, call the function recursively
        if (!isAffectedForDeletion[child]) {
            markSubtreeAffected(outDegreeValues, outDegreeIndices, outDegreePointers, dist, isAffectedForDeletion, affectedNodesForDeletion, child);
        }
    }
}


/*
1. ssspTree (done - regular 0-indexed)
2. graphCSR (done -regular 0-indexed)
3. shortestDist (done 0-indexed)
4. parentList (parent 0-indexed)
5. Predecessor (done - transposed 0-indexed)
6. Changed edges (done - transposed 0-indexed)
*/


void updateShortestPath( std::vector<int>& new_graph_values,  std::vector<int>& new_graph_column_indices,  std::vector<int>& new_graph_row_pointers, 
                         std::vector<int>& outDegreeValues,  std::vector<int>& outDegreeIndices,  std::vector<int>& outDegreePointers,
                        std::vector<int>& dist, std::vector<int>& parent , std::vector<int>& inDegreeValues, std::vector<int>& inDegreeColumnPointers, std::vector<int>& inDegreeRowValues) {

    clock_t start = std::clock();

    
    std::vector<int> t_insert_values, t_insert_row_indices, t_insert_column_pointers;
    readMTXToTransposeCSR("changed_edges.mtx", t_insert_values, t_insert_row_indices, t_insert_column_pointers, 1); // Insert mode 1
    std::vector<int> t_delete_values, t_delete_row_indices, t_delete_column_pointers;
    readMTXToTransposeCSR("changed_edges.mtx", t_delete_values, t_delete_row_indices, t_delete_column_pointers, 2); // Delete mode 2

    std::vector<int> affectedNodesList(outDegreePointers.size(), 0);
    std::vector<int> affectedNodesN(outDegreePointers.size(), 0);
    std::vector<int> affectedNodesDel(outDegreePointers.size(), 0);


    cl::sycl::queue q(cl::sycl::gpu_selector_v);

    // For insertion    
    {
        // Changed Edges
        cl::sycl::buffer t_insert_column_pointers_buf(t_insert_column_pointers.data(), cl::sycl::range<1>(t_insert_column_pointers.size()));
        cl::sycl::buffer t_insert_row_indices_buf(t_insert_row_indices.data(), cl::sycl::range<1>(t_insert_row_indices.size()));
        cl::sycl::buffer t_insert_values_buf(t_insert_values.data(), cl::sycl::range<1>(t_insert_values.size()));

        // SSSP Tree
        cl::sycl::buffer outDegreeValues_buf(outDegreeValues.data(), cl::sycl::range<1>(outDegreeValues.size()));
        cl::sycl::buffer outDegreeIndices_buf(outDegreeIndices.data(), cl::sycl::range<1>(outDegreeIndices.size()));
        cl::sycl::buffer outDegreePointers_buf(outDegreePointers.data(), cl::sycl::range<1>(outDegreePointers.size()));

        // Distance
        cl::sycl::buffer dist_buf(dist.data(), cl::sycl::range<1>(dist.size()));

        // Parent
        cl::sycl::buffer parent_buf(parent.data(), cl::sycl::range<1>(parent.size()));
        
        // AffectedNodesList
        sycl::buffer<int> affectedNodesList_buf(affectedNodesList.data(), sycl::range<1>(affectedNodesList.size()));


        q.submit([&](cl::sycl::handler& cgh) 
        {
            auto t_insert_column_pointers_acc = t_insert_column_pointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto t_insert_row_indices_acc = t_insert_row_indices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto t_insert_values_acc = t_insert_values_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            
            auto outDegreeValues_acc = outDegreeValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto outDegreeIndices_acc = outDegreeIndices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto outDegreePointers_acc = outDegreePointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            
            auto dist_acc = dist_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto parent_acc = parent_buf.get_access<cl::sycl::access::mode::read_write>(cgh);


            auto affectedNodesList_acc = affectedNodesList_buf.get_access<sycl::access::mode::read_write>(cgh);
    
            
             cgh.parallel_for<class MyKernel>(sycl::range<1>{t_insert_column_pointers_acc.size() - 1}, [=](sycl::id<1> idx) 
             {  
                // Check if (v->u) improves or not.
                int u = idx[0];
                int start = t_insert_column_pointers_acc[u];
                int end = t_insert_column_pointers_acc[u + 1];

                for (int i = start; i < end; ++i) {
                    int v = t_insert_row_indices_acc[i];
                    affectedNodesList_acc[v] = 0;
                    int alt = dist_acc[v] + t_insert_values_acc[i];
                    if (alt < dist_acc[u]) {
                        dist_acc[u] = alt;
                        parent_acc[u] = v;
                        affectedNodesList_acc[u] = 1; 
                    }
                }
            });
        });
        q.wait_and_throw();

    }

    // For deletion
    {

        cl::sycl::buffer t_delete_column_pointers_buf(t_delete_column_pointers.data(), cl::sycl::range<1>(t_delete_column_pointers.size()));
        cl::sycl::buffer t_delete_row_indices_buf(t_delete_row_indices.data(), cl::sycl::range<1>(t_delete_row_indices.size()));
        cl::sycl::buffer t_delete_values_buf(t_delete_values.data(), cl::sycl::range<1>(t_delete_values.size()));

        // SSSP Tree
        cl::sycl::buffer outDegreeValues_buf(outDegreeValues.data(), cl::sycl::range<1>(outDegreeValues.size()));
        cl::sycl::buffer outDegreeIndices_buf(outDegreeIndices.data(), cl::sycl::range<1>(outDegreeIndices.size()));
        cl::sycl::buffer outDegreePointers_buf(outDegreePointers.data(), cl::sycl::range<1>(outDegreePointers.size()));

        cl::sycl::buffer inDegreeValues_buf(inDegreeValues.data(), cl::sycl::range<1>(inDegreeValues.size()));
        cl::sycl::buffer inDegreeColumnPointers_buf(inDegreeColumnPointers.data(), cl::sycl::range<1>(inDegreeColumnPointers.size()));
        cl::sycl::buffer inDegreeRowValues_buf(inDegreeRowValues.data(), cl::sycl::range<1>(inDegreeRowValues.size()));

        // Distance
        cl::sycl::buffer dist_buf(dist.data(), cl::sycl::range<1>(dist.size()));

        // Parent
        cl::sycl::buffer parent_buf(parent.data(), cl::sycl::range<1>(parent.size()));

        // AffectedNodesList
        sycl::buffer<int> affectedNodesList_buf(affectedNodesList.data(), sycl::range<1>(affectedNodesList.size()));

        // AffectedNodesDel
        sycl::buffer<int> affectedNodesDel_buf(affectedNodesDel.data(), sycl::range<1>(affectedNodesDel.size()));

        q.submit([&](cl::sycl::handler& cgh) 
        {
            auto t_delete_column_pointers_acc = t_delete_column_pointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto t_delete_row_indices_acc = t_delete_row_indices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto t_delete_values_acc = t_delete_values_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            
            auto outDegreeValues_acc = outDegreeValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto outDegreeIndices_acc = outDegreeIndices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto outDegreePointers_acc = outDegreePointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);

            auto inDegreeValues_acc = inDegreeValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto inDegreeColumnPointers_acc = inDegreeColumnPointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto inDegreeRowValues_acc = inDegreeRowValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            
            auto dist_acc = dist_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto parent_acc = parent_buf.get_access<cl::sycl::access::mode::read_write>(cgh);


            auto affectedNodesList_acc = affectedNodesList_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto affectedNodesDel_acc = affectedNodesDel_buf.get_access<sycl::access::mode::read_write>(cgh);
    
            
             cgh.parallel_for<class MyKernel2>(sycl::range<1>{t_delete_column_pointers_acc.size() - 1}, [=](sycl::id<1> idx) 
             {
                // if (v -> u) is deleted
                int u = idx[0];

                int start = t_delete_column_pointers_acc[u];
                int end = t_delete_column_pointers_acc[u + 1];


                for (int i = start; i < end; ++i) {
                    int v = t_delete_row_indices_acc[i];
 
                    if (parent_acc[u] == v) {                        
                        affectedNodesDel_acc[u] = 1; // Mark the starting node
                        affectedNodesList_acc[u] = 1; 

                        int newDistance = INT_MAX;
                        int newParentIndex = -1;

                        int start = inDegreeColumnPointers_acc[u];
                        int end = inDegreeColumnPointers_acc[u + 1];

                        for(int i = start; i < end; ++i) {
                            int pred = inDegreeColumnPointers_acc[i]; // This is the vertex having an edge to 'u'
                            int weight = inDegreeValues_acc[i]; // This is the weight of the edge from 'vertex' to 'u'
                            
                            if(dist_acc[pred] + weight < newDistance )
                            {
                                newDistance = dist_acc[pred] + weight;
                                newParentIndex = pred; 
                            }
                            
                        }
                        
                        int oldParent = parent_acc[u];
                        if (newParentIndex == -1)
                        {
                            parent_acc[u] = -1; 
                            dist_acc[u] = INT_MAX; 
                        }
                        else
                        {
                            dist_acc[u] = newDistance;
                            parent_acc[u] = newParentIndex;
                            affectedNodesDel_acc[u] = 1;
                        }
                        
                    }
                }

            });
        });
        q.wait_and_throw();
    }

    // Find the neighbors of effected nodes
    {
        // SSSP Tree
        cl::sycl::buffer outDegreeValues_buf(outDegreeValues.data(), cl::sycl::range<1>(outDegreeValues.size()));
        cl::sycl::buffer outDegreeIndices_buf(outDegreeIndices.data(), cl::sycl::range<1>(outDegreeIndices.size()));
        cl::sycl::buffer outDegreePointers_buf(outDegreePointers.data(), cl::sycl::range<1>(outDegreePointers.size()));

        // AffectedNodesList
        sycl::buffer<int> affectedNodesList_buf(affectedNodesList.data(), sycl::range<1>(affectedNodesList.size()));
        sycl::buffer<int> affectedNodesN_buf(affectedNodesN.data(), sycl::range<1>(affectedNodesN.size()));


        q.submit([&](cl::sycl::handler& cgh) 
        {
            
            auto outDegreeValues_acc = outDegreeValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto outDegreeIndices_acc = outDegreeIndices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
            auto outDegreePointers_acc = outDegreePointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);


            auto affectedNodesList_acc = affectedNodesList_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto affectedNodesN_acc = affectedNodesN_buf.get_access<sycl::access::mode::read_write>(cgh);
            
             cgh.parallel_for<class MyKernel3>(sycl::range<1>{affectedNodesList_acc.size()}, [=](sycl::id<1> idx) 
             {
                
                int u = idx[0];
                if (affectedNodesList_acc[u] == 1)
                {
                    int start = outDegreePointers_acc[u];
                    int end = outDegreePointers_acc[u + 1];
                    for (int i = start; i < end; ++i) {
                        int v = outDegreeIndices_acc[i];
                        affectedNodesN_acc[v] = 1; 
                    }
                }
                

            });
        });
        q.wait_and_throw();

    
    clock_t end = std::clock();
    globalTiming.updateShortestPath_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    clock_t startPart2 = std::clock();

    while(1)
    {
        size_t n = affectedNodesN.size();
        vector<int> compactAffectedNodesN(affectedNodesN.size());

        // Find Compact Neighbor list
        sycl::buffer<int, 1> affectedNodesNBuf(affectedNodesN.data(), sycl::range<1>(affectedNodesN.size()));
        sycl::buffer<int, 1> compactAffectedNodesNBuf(compactAffectedNodesN.data(), sycl::range<1>(compactAffectedNodesN.size()));
        sycl::buffer<int, 1> indicesBuf(sycl::range<1>(affectedNodesN.size()));

        
        sycl::buffer<int, 1> counterBuf(sycl::range<1>(1));
        {
            auto counterAcc = counterBuf.get_access<sycl::access::mode::write>();
            counterAcc[0] = 0;
        }

        q.submit([&](sycl::handler& cgh) {
            auto affectedNodesNAcc = affectedNodesNBuf.get_access<sycl::access::mode::read>(cgh);
            auto compactAffectedNodesNAcc = compactAffectedNodesNBuf.get_access<sycl::access::mode::read_write>(cgh);
            auto indicesAcc = indicesBuf.get_access<sycl::access::mode::write>(cgh);
            auto counterAcc = counterBuf.get_access<sycl::access::mode::atomic>(cgh);

            cgh.parallel_for(sycl::range<1>(affectedNodesN.size()), [=](sycl::id<1> i) {
                if (affectedNodesNAcc[i] != 0) {
                    int index = counterAcc[0].fetch_add(1);
                    indicesAcc[index] = i;
                    compactAffectedNodesNAcc[index] = i;
                }
            });
        });

        q.wait();

        int nonZeroCount;
        {
            auto counterAcc = counterBuf.get_access<sycl::access::mode::read>();
            nonZeroCount = counterAcc[0];
        }
        auto hostIndicesAcc = indicesBuf.get_access<sycl::access::mode::read>();

        if (!nonZeroCount)
            break;
           
        {

            // SSSP Tree
            cl::sycl::buffer outDegreeValues_buf(outDegreeValues.data(), cl::sycl::range<1>(outDegreeValues.size()));
            cl::sycl::buffer outDegreeIndices_buf(outDegreeIndices.data(), cl::sycl::range<1>(outDegreeIndices.size()));
            cl::sycl::buffer outDegreePointers_buf(outDegreePointers.data(), cl::sycl::range<1>(outDegreePointers.size()));

            cl::sycl::buffer inDegreeValues_buf(inDegreeValues.data(), cl::sycl::range<1>(inDegreeValues.size()));
            cl::sycl::buffer inDegreeColumnPointers_buf(inDegreeColumnPointers.data(), cl::sycl::range<1>(inDegreeColumnPointers.size()));
            cl::sycl::buffer inDegreeRowValues_buf(inDegreeRowValues.data(), cl::sycl::range<1>(inDegreeRowValues.size()));

            // Distance
            cl::sycl::buffer dist_buf(dist.data(), cl::sycl::range<1>(dist.size()));

            // Parent
            cl::sycl::buffer parent_buf(parent.data(), cl::sycl::range<1>(parent.size()));

            // AffectedNodesList
            sycl::buffer<int> affectedNodesList_buf(affectedNodesList.data(), sycl::range<1>(affectedNodesList.size()));
            sycl::buffer<int> affectedNodesN_buf(affectedNodesN.data(), sycl::range<1>(affectedNodesN.size()));
            sycl::buffer<int> compactAffectedNodesN_buf(compactAffectedNodesN.data(), sycl::range<1>(nonZeroCount));

            // AffectedNodesDel
            sycl::buffer<int> affectedNodesDel_buf(affectedNodesDel.data(), sycl::range<1>(affectedNodesDel.size()));
            sycl::buffer<int> frontierSize_buf(&nonZeroCount, cl::sycl::range<1>(1));


            q.submit([&](cl::sycl::handler& cgh) 
            {
                
                auto outDegreeValues_acc = outDegreeValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                auto outDegreeIndices_acc = outDegreeIndices_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                auto outDegreePointers_acc = outDegreePointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                
                auto inDegreeValues_acc = inDegreeValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                auto inDegreeColumnPointers_acc = inDegreeColumnPointers_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                auto inDegreeRowValues_acc = inDegreeRowValues_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                
                auto dist_acc = dist_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                auto parent_acc = parent_buf.get_access<cl::sycl::access::mode::read_write>(cgh);


                auto affectedNodesList_acc = affectedNodesList_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto affectedNodesN_acc = affectedNodesN_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto affectedNodesDel_acc = affectedNodesDel_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto compactAffectedNodesN_acc = compactAffectedNodesN_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto hostIndicesAcc = indicesBuf.get_access<sycl::access::mode::read>();
                auto counterAcc = counterBuf.get_access<sycl::access::mode::read>();
                auto frontierSize_acc = frontierSize_buf.get_access<sycl::access::mode::read>();
                



                cgh.parallel_for<class MyKernel4>(sycl::range<1>{frontierSize_acc.size()}, [=](sycl::id<1> idx) 
                {   
                    //propagate [A(v)->n]
                    int n = hostIndicesAcc[idx[0]];


                    affectedNodesN_acc[n] = 0; 
                    int start = inDegreeColumnPointers_acc[n];
                    int end = inDegreeColumnPointers_acc[n + 1];

                    int newDistance = INT_MAX;
                    int newParentIndex = -1; 

                    int flag = 0;
                    // Scan for each potential parents
                    for (int i = start; i < end; ++i) {

                        int v = inDegreeRowValues_acc[i];

                        //Avoid deletion of root 
                        if (n + 1 == 1)
                            continue;


                        int pred = inDegreeColumnPointers_acc[i]; 
                        int weight = v; 
                        
                        if(dist_acc[pred] + weight != newDistance )
                        {
                            if (INT_MAX - weight >= newDistance)
                            {
                                // Infinite Loop: Edge case detected
                                return;
                            }
                            newDistance = dist_acc[pred] + weight;
                            newParentIndex = pred; 
                            flag = 1; 
                        }
                            
                        
                        int oldParent = parent_acc[n];
                        if (newParentIndex == -1)
                        {
                            parent_acc[n] = -1; 
                            dist_acc[n] = INT_MAX; 
                        }
                        else
                        {
                            dist_acc[n] = newDistance;
                            parent_acc[n] = newParentIndex;
                        }

                        if (flag == 1)
                        {
                            int start = outDegreePointers_acc[n];
                            int end = outDegreePointers_acc[n + 1]; 

                            for (int i = start; i < end; ++i) {

                                affectedNodesN_acc[outDegreeIndices_acc[i]] = 1;
                            }
                        }
                    }

                });
            });
            q.wait_and_throw();
        }
    }

    clock_t endPart2 = std::clock();
    // Note: We're only timing the first part for updateShortestPath

    }

}


void dijkstra(const std::vector<int>& values, const std::vector<int>& column_indices, const std::vector<int>& row_pointers, int src, std::vector<int>& dist, std::vector<int>& parent){
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

    // Add the last row_pointer, why?
    row_pointers[current_row + 1] = nnz;

    // Close the file
    file.close();

    return true;
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

    // Initialize timing variables
    globalTiming.updateShortestPath_time = 0.0;
    globalTiming.constructGraph_time = 0.0;
    globalTiming.updateWeights_time = 0.0;
    globalTiming.parallelBellmanFord_time = 0.0;
    globalTiming.total_time = 0.0;

    clock_t total_start = clock();

//Input Graph to CSR
    // road-road-usa.mtx, rgg_n_2_20_s0.mtx, 
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

    std::vector<int> outDegreeValues;
    std::vector<int> outDegreeIndices;
    std::vector<int> outDegreePointers;
    sortAndSaveMTX("SSSP_Tree.mtx", "sorted_SSSP_Tree.mtx");
    readMTXToCSR("sorted_graph.mtx", outDegreeValues, outDegreeIndices, outDegreePointers);
    //printCSRRepresentation(outDegreeValues, outDegreeIndices, outDegreePointers);
// Changed edges
    auto originalGraph = readMTX("sorted_graph.mtx");

    int numVertices = row_pointers.size() - 1;  // Should be determined from the MTX file or another source
    int numChanges = 5;
    int minWeight = 1;
    int maxWeight = 10;

    std::vector<std::tuple<int, int, int>> changedEdges;
    float deletePercentage = 1.0f;
    auto newGraph = generateChangedGraph(originalGraph, numVertices, numChanges, minWeight, maxWeight, changedEdges, deletePercentage);
    // writeMTX by default sort by row for easy readings
    writeMTX("new_graph.mtx", newGraph, numVertices, true); 
    writeMTX("changed_edges.mtx", changedEdges, numVertices, false);

    std::vector<int> new_graph_values;
    std::vector<int> new_graph_column_indices;
    std::vector<int> new_graph_row_pointers;
    readMTXToCSR("new_graph.mtx", new_graph_values, new_graph_column_indices, new_graph_row_pointers);
    //printCSRRepresentation(new_graph_values, new_graph_column_indices, new_graph_row_pointers);


    // Find Predecessor
    std::vector<int> inDegreeValues;
    std::vector<int> inDegreeColumnPointers;
    std::vector<int> inDegreeRowValues;
    readMTXToTransposeCSR("new_graph.mtx", inDegreeValues, inDegreeRowValues, inDegreeColumnPointers);
    //printCSRRepresentation(inDegreeValues, inDegreeRowValues, inDegreeColumnPointers);

    

    updateShortestPath(new_graph_values, new_graph_column_indices, new_graph_row_pointers, outDegreeValues, outDegreeIndices, outDegreePointers, dist, parent, inDegreeValues, inDegreeColumnPointers, inDegreeRowValues);

//Tree 2

    sortAndSaveMTX("graph.mtx", "sorted_graph2.mtx");

    std::vector<int> values2;
    std::vector<int> column_indices2;
    std::vector<int> row_pointers2;

    readMTXToCSR("sorted_graph2.mtx", values2, column_indices2, row_pointers2);
    //printCSRRepresentation(values, column_indices, row_pointers);

//Find SSSP tree and store in mtx file
    std::vector<int> parent2(row_pointers2.size() - 1, -1); // takes child and returns it's parent
    std::vector<int> dist2(row_pointers2.size() - 1, INT_MAX);
    
    // Run Dijkstra's algorithm from source vertex 0
    dijkstra(values2, column_indices2, row_pointers2, 0, dist2, parent2);

    saveSSSPTreeToFile("SSSP_Tree2.mtx", values2, column_indices2, row_pointers2, parent2);

    std::vector<int> outDegreeValues2;
    std::vector<int> outDegreeIndices2;
    std::vector<int> outDegreePointers2;
    sortAndSaveMTX("SSSP_Tree2.mtx", "sorted_SSSP_Tree2.mtx");
    readMTXToCSR("sorted_graph2.mtx", outDegreeValues2, outDegreeIndices2, outDegreePointers2);
    //printCSRRepresentation(outDegreeValues, outDegreeIndices, outDegreePointers);
// Changed edges
    auto originalGraph2 = readMTX("sorted_graph2.mtx");

    int numVertices2 = row_pointers2.size() - 1;  // Should be determined from the MTX file or another source
    int numChanges2 = 5;
    int minWeight2 = 1;
    int maxWeight2 = 10;

    std::vector<std::tuple<int, int, int>> changedEdges2;
    float deletePercentage2 = 1.0f;
    auto newGraph2 = generateChangedGraph(originalGraph2, numVertices2, numChanges2, minWeight2, maxWeight2, changedEdges2, deletePercentage2);
    // writeMTX by default sort by row for easy readings
    writeMTX("new_graph2.mtx", newGraph2, numVertices2, true); 
    writeMTX("changed_edges2.mtx", changedEdges2, numVertices2, false);

    std::vector<int> new_graph_values2;
    std::vector<int> new_graph_column_indices2;
    std::vector<int> new_graph_row_pointers2;
    readMTXToCSR("new_graph2.mtx", new_graph_values2, new_graph_column_indices2, new_graph_row_pointers2);
    //printCSRRepresentation(new_graph_values, new_graph_column_indices, new_graph_row_pointers);


    // Find Predecessor
    std::vector<int> inDegreeValues2;
    std::vector<int> inDegreeColumnPointers2;
    std::vector<int> inDegreeRowValues2;
    readMTXToTransposeCSR("new_graph2.mtx", inDegreeValues2, inDegreeRowValues2, inDegreeColumnPointers2);
    //printCSRRepresentation(inDegreeValues, inDegreeRowValues, inDegreeColumnPointers);

    

    updateShortestPath(new_graph_values2, new_graph_column_indices2, new_graph_row_pointers2, outDegreeValues2, outDegreeIndices2, outDegreePointers2, dist2, parent2, inDegreeValues2, inDegreeColumnPointers2, inDegreeRowValues2);

// Tree 3


    sortAndSaveMTX("graph.mtx", "sorted_graph3.mtx");

    std::vector<int> values3;
    std::vector<int> column_indices3;
    std::vector<int> row_pointers3;

    readMTXToCSR("sorted_graph3.mtx", values3, column_indices3, row_pointers3);
    //printCSRRepresentation(values, column_indices, row_pointers);

//Find SSSP tree and store in mtx file
    std::vector<int> parent3(row_pointers3.size() - 1, -1); // takes child and returns it's parent
    std::vector<int> dist3(row_pointers3.size() - 1, INT_MAX);
    
    // Run Dijkstra's algorithm from source vertex 0
    dijkstra(values3, column_indices3, row_pointers3, 0, dist3, parent3);

    saveSSSPTreeToFile("SSSP_Tree3.mtx", values3, column_indices3, row_pointers3, parent3);

    std::vector<int> outDegreeValues3;
    std::vector<int> outDegreeIndices3;
    std::vector<int> outDegreePointers3;
    sortAndSaveMTX("SSSP_Tree3.mtx", "sorted_SSSP_Tree3.mtx");
    readMTXToCSR("sorted_graph3.mtx", outDegreeValues3, outDegreeIndices3, outDegreePointers3);
    //printCSRRepresentation(outDegreeValues, outDegreeIndices, outDegreePointers);
// Changed edges
    auto originalGraph3 = readMTX("sorted_graph3.mtx");

    int numVertices3 = row_pointers3.size() - 1;  // Should be determined from the MTX file or another source
    int numChanges3 = 5;
    int minWeight3 = 1;
    int maxWeight3 = 10;

    std::vector<std::tuple<int, int, int>> changedEdges3;
    float deletePercentage3 = 1.0f;
    auto newGraph3 = generateChangedGraph(originalGraph3, numVertices3, numChanges3, minWeight3, maxWeight3, changedEdges3, deletePercentage3);
    // writeMTX by default sort by row for easy readings
    writeMTX("new_graph3.mtx", newGraph3, numVertices3, true); 
    writeMTX("changed_edges3.mtx", changedEdges3, numVertices3, false);

    std::vector<int> new_graph_values3;
    std::vector<int> new_graph_column_indices3;
    std::vector<int> new_graph_row_pointers3;
    readMTXToCSR("new_graph3.mtx", new_graph_values3, new_graph_column_indices3, new_graph_row_pointers3);
    //printCSRRepresentation(new_graph_values, new_graph_column_indices, new_graph_row_pointers);


    // Find Predecessor
    std::vector<int> inDegreeValues3;
    std::vector<int> inDegreeColumnPointers3;
    std::vector<int> inDegreeRowValues3;
    readMTXToTransposeCSR("new_graph3.mtx", inDegreeValues3, inDegreeRowValues3, inDegreeColumnPointers3);
    //printCSRRepresentation(inDegreeValues, inDegreeRowValues, inDegreeColumnPointers);

    

    updateShortestPath(new_graph_values3, new_graph_column_indices3, new_graph_row_pointers3, outDegreeValues3, outDegreeIndices3, outDegreePointers3, dist3, parent3, inDegreeValues3, inDegreeColumnPointers3, inDegreeRowValues3);


// Create trees
    
    
    Tree1 t1(parent);
    Tree1 t2(parent2);
    Tree1 t3(parent3);
    std::vector<Tree*> trees = {&t1, &t2, &t3};

    int k = trees.size();

// Preference 
    std::vector<int> Pref = {1, 1, 1};

    clock_t construct_start = clock();
    std::vector<Edge> edges = constructGraph(trees, 3);
    clock_t construct_end = clock();
    globalTiming.constructGraph_time = ((double)(construct_end - construct_start)) / CLOCKS_PER_SEC;
    
    // for (const auto& edge : edges) {
    //     std::cout << "Edge from " << edge.u << " to " << edge.v << " with weight " << edge.weight << std::endl;
    // }

    std::vector<int> weights(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
        weights[i] = edges[i].weight;
    }
    updateWeights(edges, Pref, k);

    std::vector<int> finalParents;
    parallelBellmanFord(edges, numVertices, 0, finalParents);
    
    // Calculate total time
    clock_t total_end = clock();
    globalTiming.total_time = ((double)(total_end - total_start)) / CLOCKS_PER_SEC;
    
    // Write timing results to file
    std::ofstream timingFile("timing_results.txt");
    if (timingFile.is_open()) {
        timingFile << "MOSP Algorithm Timing Results\n";
        timingFile << "==============================\n\n";
        timingFile << "Individual Function Timings:\n";
        timingFile << "updateShortestPath(): " << globalTiming.updateShortestPath_time << " seconds\n";
        timingFile << "constructGraph(): " << globalTiming.constructGraph_time << " seconds\n";
        timingFile << "updateWeights(): " << globalTiming.updateWeights_time << " seconds\n";
        timingFile << "parallelBellmanFord(): " << globalTiming.parallelBellmanFord_time << " seconds\n";
        timingFile << "\nTotal Execution Time: " << globalTiming.total_time << " seconds\n";
        timingFile.close();
        std::cout << "Timing results written to timing_results.txt" << std::endl;
    } else {
        std::cerr << "Failed to open timing_results.txt for writing" << std::endl;
    }
    
    return 0;
}

// To run, clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda SYCL_Final.cpp -o SYCL_Final && ./SYCL_Final

    
