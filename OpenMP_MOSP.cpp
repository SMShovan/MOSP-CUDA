#include <bits/stdc++.h>
#include <fstream>
#include <ctime>
int NUM_THREADS = 4;
using namespace std;

// Timing structure to store all timing data
struct TimingData {
    double updateShortestPath_time;
    double constructGraph_time;
    double bellmanFord_time;
    double total_time;
};

// Global timing data
TimingData globalTiming;

struct Edge {
    int source;
    int destination;
    double weight;
    Edge(int src, int dest, double w) : source(src), destination(dest), weight(w) {}
};

using Graph = std::unordered_map<int, std::unordered_map<int, int>>;
int& access_with_default(Graph& g, int key1, int key2, int default_value = 4) {
    if(g[key1].find(key2) == g[key1].end()) {
        g[key1][key2] = default_value;
    }
    return g[key1][key2];
}
class Tree {
public:
    virtual const std::vector<int>& getParents() const = 0;  // Pure virtual method
    virtual ~Tree() = default;  // Virtual destructor
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

void addEdge(Graph& graph, int u, int v, double pref) {
    #pragma omp critical
    {
        access_with_default(graph, u, v) = access_with_default(graph, u, v) - 1.0/pref;
    }
}

Graph constructGraph(const std::vector<Tree*>& trees, const std::vector<double>& Pref) {
    clock_t start = clock();
    
    Graph graph;

    for (size_t index = 0; index < trees.size(); ++index) {
        const auto& tree = trees[index];
        const std::vector<int>& parents = tree->getParents();

        #pragma omp parallel for
        for (int i = 1; i < parents.size(); ++i) {
            if (parents[i] != 0) {
                addEdge(graph, parents[i], i, Pref[index]);
            }
        }
    }
    
    clock_t end = clock();
    globalTiming.constructGraph_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    return graph;
}

bool bellmanFord(const Graph& graph, int source, std::unordered_map<int, int>& distances, std::vector<int>& newParent) {
    clock_t start = clock();
    
    int V = graph.size();
    newParent.assign(V + 1, -1);  // Initialize with -1 (no predecessor)

    // Initialize distances
    for (const auto& [node, _] : graph) {
        distances[node+1] = std::numeric_limits<double>::infinity();
    }
    distances[source] = 0;

    // Relax edges |V| - 1 times
    for (size_t i = 1; i <= V; ++i) {
        #pragma omp parallel for
        for (size_t j = 0; j < graph.size(); ++j) {
            auto it = std::next(graph.begin(), j);
            int u = it->first;
            const auto& neighbors = it->second;
            for (const auto& [v, weight] : neighbors) {
                if (!std::isinf(distances[u]) && distances[u] + weight < distances[v]) {
                    #pragma omp critical
                    {
                        if (distances[u] + weight < distances[v]) {
                            distances[v] = distances[u] + weight;
                            newParent[v] = u;
                        }
                    }
                }
            }
        }
    }


    // Check for negative-weight cycles
    for (const auto& [u, neighbors] : graph) {
        for (const auto& [v, weight] : neighbors) {
            if (!std::isinf(distances[u]) && distances[u] + weight < distances[v]) {
                return false;  // Graph contains a negative-weight cycle
            }
        }
    }

    clock_t end = clock();
    globalTiming.bellmanFord_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    return true;  // No negative-weight cycles detected
}

bool operator==(const Edge& lhs, const Edge& rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination && lhs.weight == rhs.weight;
}

std::vector<std::vector<Edge>> convertToDT(std::ifstream& inputFile, bool isGraph, std::vector<std::vector<Edge>>& predecessor) {
    std::string line;
    int numRows, numCols, numNonZero;
    if (!std::getline(inputFile, line)) {
        std::cerr << "Error: Unable to read the header line from the input file." << std::endl;
        exit(1);
    }
    std::istringstream headerStream(line);
    if (!(headerStream >> numRows >> numCols >> numNonZero)) {
        std::cerr << "Error: Invalid header format in the input file." << std::endl;
        exit(1);
    }
    std::vector<std::vector<Edge>> DTMatrix(numRows);
    int lineCount = 1;
    while (std::getline(inputFile, line)) {
        lineCount++;
        std::istringstream lineStream(line);
        int row, col;
        double value;
        if (!(lineStream >> row >> col >> value)) {
            std::cerr << "Error: Invalid line format at line " << lineCount << " in the input file." << std::endl;
            exit(1);
        }
        if (row < 1 || row > numRows || col < 1 || col > numCols) {
            std::cerr << "Error: Invalid vertex indices at line " << lineCount << " in the input file." << std::endl;
            exit(1);
        }
        DTMatrix[row - 1].emplace_back(row - 1, col - 1, value);
        if(isGraph)
        {
            predecessor.resize(numCols);
            predecessor[col - 1].emplace_back(row, col, value);
        } 
    }
    return DTMatrix;
}

std::vector<std::vector<int>> dijkstra(const std::vector<std::vector<Edge>>& graphDT, int sourceNode, std::vector<double>& shortestDist, std::vector<int>& parentList) {
    int numNodes = graphDT.size();
    std::vector<bool> visited(numNodes, false);
    shortestDist[sourceNode - 1] = 0;
    for (int i = 0; i < numNodes - 1; ++i) {
        int minDistNode = -1;
        double minDist = std::numeric_limits<double>::infinity();
        for (int j = 0; j < numNodes; ++j) {
            if (!visited[j] && shortestDist[j] < minDist) {
                minDist = shortestDist[j];
                minDistNode = j;
            }
        }
        if (minDistNode == -1) {
            break;
        }
        visited[minDistNode] = true;
        for (const Edge& edge : graphDT[minDistNode]) {
            int neighbor = edge.destination;
            double weight = edge.weight;
            if (!visited[neighbor] && shortestDist[minDistNode] + weight < shortestDist[neighbor]) {
                shortestDist[neighbor] = shortestDist[minDistNode] + weight;
            }
        }
    }
    std::vector<std::vector<int>> ssspTree(numNodes);
    std::vector<bool> cycleCheck (numNodes, false);
    for (int i = 0; i < numNodes; ++i) {
        if (shortestDist[i] != std::numeric_limits<double>::infinity()) {
            int parent = i + 1;
            for (const Edge& edge : graphDT[i]) {
                int child = edge.destination + 1;
                if (shortestDist[child - 1] == shortestDist[i] + edge.weight && !cycleCheck[child - 1]) {
                    ssspTree[parent - 1].push_back(child);
                    cycleCheck[child - 1 ] = true;
                    parentList[child] = parent;
                }
            }
        }
    }
    return ssspTree;
}

void printShortestPathTree(const std::vector<std::pair<int, std::vector<int>>>& parentChildSSP) {
    std::cout << "Shortest Path Tree:\n";
    for (const auto& node : parentChildSSP) {
        std::cout << "Node " << node.first << ": ";
        for (int child : node.second) {
            std::cout << child << " ";
        }
        std::cout << "\n";
    }
}


void markSubtreeAffected(std::vector<std::pair<int, std::vector<int>>>& parentChildSSP, std::vector<double>& shortestDist, std::vector<bool>& affectedNodes, std::queue<int>& affectedNodesQueue, std::vector<bool>& affectedDelNodes, int node, std::vector<int>& affectedNodesList, std::vector<int>& affectedNodesDel) {

    #pragma omp parallel for
    for (size_t i = 0; i < parentChildSSP[node].second.size(); ++i) {
        int child = parentChildSSP[node].second[i];
        if( !affectedDelNodes[child - 1])
        {
            affectedDelNodes[child - 1] = true;
            affectedNodesList[child - 1] = 1; 
            affectedNodesDel[child - 1] = 1;
        }
    }
}

std::vector<std::pair<int, std::vector<int>>> convertToParentChildSSP(const std::vector<std::vector<int>>& ssspTree) {
    std::vector<std::pair<int, std::vector<int>>> parentChildSSP(ssspTree.size());
    for (int i = 0; i < ssspTree.size(); ++i) {
        parentChildSSP[i].first = i + 1;  
        parentChildSSP[i].second = ssspTree[i]; 
    }
    return parentChildSSP;
}


/*
1. ssspTree is a subset (0-indexed) of graphDT having the edges belongs to shortest path. However, ssspTree contains a set of pairs with parent (1-indexed) and children (1-indexed).
2. graphDT is the whole graph containg a set of vector of edges. It is 0-indexed. i index contains all the outgoing edges (also 0-indexed) from vertex (i+1). The graph is assumed to have vertex id > 0. 
3. shortestDist is the set of all vertices (0-indexed) from the source node assumed to be 1 (in 1-index).
4. parentList is the set of all parents (1-indexed) given the child node (1-indexed) of the ssspTree.
5. Global variable Predecessor is similar to graphDT, however, instead of storing into rows, it stores in column index to find all incident vertices. Again the column index is 0-indexed, but edges are 1-indexed.
*/

void updateShortestPath(std::vector<std::pair<int, std::vector<int>>>& ssspTree, std::vector<std::vector<Edge>>& graphDT, const std::vector<std::vector<Edge>>& changedEdgesDT, std::vector<double>& shortestDist, std::vector<int>& parentList, std::vector<std::vector<Edge>>& predecessor) {
    std::vector<Edge> insertedEdges;
    std::vector<Edge> deletedEdges;
    std::vector<bool> affectedNodes(ssspTree.size(), false);
    std::vector<bool> affectedDelNodes(ssspTree.size(), false);
    std::queue<int> affectedNodesQueue;
    std::vector<int> affectedNodesList(ssspTree.size(), 0);
    std::vector<int> affectedNodesN(ssspTree.size(), 0);
    std::vector<int> affectedNodesDel(ssspTree.size(), 0);
    

    std::unordered_map<int, std::vector<Edge>> insertedEdgeMap;
    std::unordered_map<int, std::vector<Edge>> deletedEdgeMap;
    std::unordered_map<int, std::vector<Edge>> affectedMap;
    

    // Identify inserted and deleted edges
    for (int i = 0; i < changedEdgesDT.size(); i++) {
        const std::vector<Edge>& row = changedEdgesDT[i];
        for (int j = 0; j < row.size(); j++) {
            const Edge& edge = row[j];
            if (edge.weight > 0) {
                insertedEdges.push_back(edge);
                insertedEdgeMap[edge.destination].push_back(edge);
                graphDT[edge.source].push_back(Edge(edge.source, edge.destination, edge.weight));
                predecessor[edge.destination].emplace_back(edge.source, edge.destination, edge.weight);
            } else if (edge.weight < 0) {
                deletedEdges.push_back(edge);
                deletedEdgeMap[edge.destination].push_back(edge);
                for (int i = 0; i < graphDT[edge.source].size(); ++i) {
                    if (graphDT[edge.source][i].source == edge.source && graphDT[edge.source][i].destination == edge.destination) {
                        graphDT[edge.source].erase(graphDT[edge.source].begin() + i);
                        break;
                    }
                }
                for (auto& preEdge : predecessor[edge.destination]) 
                {
                    if (edge.source + 1 == preEdge.source) {
                        auto it = std::find(predecessor[edge.destination].begin(), predecessor[edge.destination].end(), preEdge);
                        if (it != predecessor[edge.destination].end()) 
                        {
                            predecessor[edge.destination].erase(it);
                        }
                    }
                }
            }
        }
    }

    clock_t start = clock();

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < insertedEdges.size(); i++) {
        const Edge& edge = insertedEdges[i];
        int x,y; 
        x = edge.source;
        y = edge.destination;
        if (shortestDist[y] > shortestDist[x] + edge.weight) {
            shortestDist[y] = shortestDist[x] + edge.weight;
            
            int oldParent = parentList[y + 1];
            // for (int j = 0; j < ssspTree[oldParent - 1].second.size(); j++ )
            // {
                
            //     if (ssspTree[oldParent - 1].second[j] == y + 1 )
            //     {
            //         #pragma omp critical
            //         {
            //             ssspTree[oldParent - 1].second.erase(ssspTree[oldParent - 1].second.begin() + j);
            //         }
            //     }
            // }
            parentList[y+1] = x + 1;
            #pragma omp critical
            {
                ssspTree[x].second.push_back(y+1);
            }
            affectedNodesList[y] = 1;
        }
    }


    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < deletedEdges.size(); i++) {
        const Edge& edge = deletedEdges[i];
        bool inTree = false;
        for (auto& preEdge : predecessor[edge.destination]) 
        {
            if (edge.source + 1 == preEdge.source) {
                
                for (int i = 0; i < ssspTree[edge.source].second.size() ; i++)
                {
                    if (ssspTree[edge.source].second[i] == edge.destination + 1)
                    inTree = true;
                }
            }
        }
        if (!inTree)
            continue;
        
        affectedNodes[edge.destination] = true;
        affectedNodesQueue.push(edge.destination);
        affectedNodesList[edge.destination] = 1; 
        affectedDelNodes[edge.destination] = true;
        affectedNodesDel[edge.destination] = 1;
        shortestDist[edge.destination] = std::numeric_limits<double>::infinity();
        markSubtreeAffected(ssspTree, shortestDist, affectedNodes, affectedNodesQueue, affectedDelNodes, edge.destination, affectedNodesList, affectedNodesDel);

        int n = edge.destination; 
        //std::cout<<"\nPredecessor of "<< n + 1 <<" ";
        int newDistance = std::numeric_limits<double>::infinity();
        int newParentIndex = -1; 

        for (int i = 0; i < predecessor[n].size(); i++)
        {
            if (shortestDist[predecessor[n][i].source - 1] + predecessor[n][i].weight < newDistance)
            {
                newDistance = shortestDist[predecessor[n][i].source - 1] + predecessor[n][i].weight;
                newParentIndex = predecessor[n][i].source - 1;
            } 
        }
        int oldParent = parentList[edge.destination + 1];
        if (newParentIndex == -1)
        {
            parentList[n + 1] = -1; 
            shortestDist[n] = std::numeric_limits<double>::infinity();
        }   
        else 
        {    
            shortestDist[n] = newDistance; 
            ssspTree[newParentIndex].second.push_back(n+1);
            parentList[edge.destination + 1] = newParentIndex + 1;      
        }
        // for (int j = 0; j < ssspTree[oldParent - 1].second.size() && oldParent >= -1 ; j++ )
        // {
        //     if (ssspTree[oldParent - 1].second[j] == edge.destination + 1 )
        //     {
        //         #pragma omp critical
        //         {
        //             ssspTree[oldParent - 1].second.erase(ssspTree[oldParent - 1].second.begin() + j);
        //         }
        //     }
        // }
    }

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0 ; i < affectedNodesList.size(); i++)
    {
        if(affectedNodesList[i] == 1)
        {
            for (int j = 0; j < ssspTree[i].second.size(); j++)
            {
                affectedNodesN[ssspTree[i].second[j] - 1] = 1; 
            }
        }
    }

    clock_t end = clock();
    globalTiming.updateShortestPath_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    clock_t startPart2 = clock();

    while (1)
    {
        int sum = 0;
        #pragma omp parallel for reduction(+:sum) num_threads(NUM_THREADS)
        for (size_t i = 0; i < affectedNodesN.size(); i++) {
            sum += affectedNodesN[i];
        }
        if(!sum)
            break;
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int n = 0 ; n < affectedNodesN.size(); n++)
        {
            if (affectedNodesN[n] == 1)
            {   
                affectedNodesN[n] = 0;
                
                int newDistance = std::numeric_limits<double>::infinity();
                int newParentIndex = -1;

                int flag = 0; 

                for (int j = 0; j < predecessor[n].size(); j++)
                {
                    

                    if (n == 0)
                        continue;
                    
                    int pred = predecessor[n][j].source;
                    int weight = predecessor[n][j].weight;

                    if (shortestDist[predecessor[n][j].source - 1] + predecessor[n][j].weight != newDistance)
                    {
                        newDistance = shortestDist[predecessor[n][j].source - 1] + predecessor[n][j].weight;
                        newParentIndex = pred;
                        flag = 1;

                        if (std::isinf(newDistance))
                        {
                            // Infinity loop detected
                            return;
                        }

                        int oldParent = parentList[n + 1];
                        if (newParentIndex == -1)
                        {
                            parentList[n + 1] = -1;
                            shortestDist[n] = std::numeric_limits<double>::infinity();
                        }
                        else
                        {
                            parentList[n + 1] = newParentIndex;
                            shortestDist[n] = newDistance;
                            

                            ssspTree[newParentIndex].second.push_back(n+1);
                            // for (int i = 0; i < ssspTree[oldParent - 1].second.size(); i++ )
                            // {
                            //     if (ssspTree[oldParent - 1].second[i] == n + 1 )
                            //     {
                            //         #pragma omp critical
                            //         {
                            //             ssspTree[oldParent - 1].second.erase(ssspTree[oldParent - 1].second.begin() + i);
                            //         }                 
                            //     }
                            // }


                        }
                        if (flag == 1)
                        {
                            for (int k = 0; k < ssspTree[n].second.size(); k++)
                            {
                                affectedNodesN[ssspTree[n].second[k] - 1] = 1;
                            } 
                        }
                    }

                }
                

            }
        }
    }
    clock_t endPart2 = clock();
    // Note: We're only timing the first part for updateShortestPath
}
    
int main(int argc, char** argv) {

    // Initialize timing variables
    globalTiming.updateShortestPath_time = 0.0;
    globalTiming.constructGraph_time = 0.0;
    globalTiming.bellmanFord_time = 0.0;
    globalTiming.total_time = 0.0;

    clock_t total_start = clock();

    if (argc < 6) {
        std::cerr << "Usage: ./program -g <graph_file> -c <changed_edges_file> -s <source_node>\n";
        return 1;
    }
    std::string graphFile;
    std::string changedEdgesFile;
    int sourceNode;
    
    for (int i = 1; i < argc; i += 2) {
        std::string option(argv[i]);
        std::string argument(argv[i + 1]);

        if (option == "-g") {
            graphFile = argument;
        } else if (option == "-c") {
            changedEdgesFile = argument;
        } else if (option == "-s") {
            sourceNode = std::stoi(argument);
        }else if (option == "-t") {
            NUM_THREADS = std::stoi(argument);
        } else {
            std::cerr << "Invalid option: " << option << "\n";
            return 1;
        }
    }
    std::ifstream graphInputFile(graphFile);
    if (!graphInputFile.is_open()) {
        std::cerr << "Unable to open graph file: " << graphFile << std::endl;
        return 1;
    }
    std::vector<std::vector<Edge>> predecessor;
    std::vector<std::vector<Edge>> graphDT = convertToDT(graphInputFile, true, predecessor);
    graphInputFile.close();
    std::vector<double> shortestDist(graphDT.size(), std::numeric_limits<double>::infinity());
    shortestDist[sourceNode - 1] = 0;
    std::vector<int> parent(graphDT.size() + 1, -1);
    std::vector<std::vector<int>> ssspTree = dijkstra(graphDT, sourceNode, shortestDist, parent);
    std::vector<std::pair<int, std::vector<int>>> parentChildSSP = convertToParentChildSSP(ssspTree);

    std::ifstream changedEdgesInputFile(changedEdgesFile);
    if (!changedEdgesInputFile.is_open()) {
        std::cerr << "Unable to open changed edges file: " << changedEdgesFile << std::endl;
        return 1;
    }
    std::vector<std::vector<Edge>> changedEdgesDT = convertToDT(changedEdgesInputFile, false, predecessor);
    changedEdgesInputFile.close();
    updateShortestPath(parentChildSSP, graphDT, changedEdgesDT, shortestDist, parent, predecessor);

    printShortestPathTree(parentChildSSP);

    Tree1 t1(parent);
    Tree2 t2(parent);
    Tree3 t3(parent);


    std::vector<Tree*> trees = {&t1, &t2, &t3};

    std::vector<double> Pref = {1.0,1.0,1.0};

    Graph graph = constructGraph(trees, Pref);

    // Print the graph
    for (const auto& [node, neighbors] : graph) {
        std::cout << "Node " << node << " has directed edges to:\n";
        for (const auto& [neighbor, weight] : neighbors) {
            std::cout << "  Node " << neighbor << " with weight " << weight << "\n";
        }
    }
    std::unordered_map<int, int> distances;
    std::vector<int> newParent;
    int source = 1;  // Example source node
    bool success = bellmanFord(graph, source, distances, newParent);

    if (success) {
        std::cout << "Shortest distances from node " << source << ":\n";
        for (const auto& [node, distance] : distances) {
            std::cout << "Node " << node << ": " << distance << ", Parent: " << newParent[node] << "\n";
        }
    } else {
        std::cout << "The graph contains a negative-weight cycle.\n";
    }

    // Calculate total time
    clock_t total_end = clock();
    globalTiming.total_time = ((double)(total_end - total_start)) / CLOCKS_PER_SEC;
    
    // Write timing results to file
    std::ofstream timingFile("timing_results_openmp.txt");
    if (timingFile.is_open()) {
        timingFile << "OpenMP MOSP Algorithm Timing Results\n";
        timingFile << "====================================\n\n";
        timingFile << "Individual Function Timings:\n";
        timingFile << "updateShortestPath(): " << globalTiming.updateShortestPath_time << " seconds\n";
        timingFile << "constructGraph(): " << globalTiming.constructGraph_time << " seconds\n";
        timingFile << "bellmanFord(): " << globalTiming.bellmanFord_time << " seconds\n";
        timingFile << "\nTotal Execution Time: " << globalTiming.total_time << " seconds\n";
        timingFile << "Number of OpenMP Threads: " << NUM_THREADS << "\n";
        timingFile.close();
        std::cout << "Timing results written to timing_results_openmp.txt" << std::endl;
    } else {
        std::cerr << "Failed to open timing_results_openmp.txt for writing" << std::endl;
    }

    return 0;
}

