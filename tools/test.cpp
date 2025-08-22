#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

bool readMTXToTransposeCSRUndirected(const std::string& filename, 
                                     std::vector<int>& values, 
                                     std::vector<int>& row_indices, 
                                     std::vector<int>& col_pointers) 
{
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

        // Insert both (i,j) and (j,i) for undirected graphs
        transposedEntries.insert({col, {row, val}});
        if(col != row) // Avoid adding duplicates for self-loops
            transposedEntries.insert({row, {col, val}});
    }

    file.close();

    values.clear();
    row_indices.clear();
    col_pointers.clear();

    int current_col = -1;
    for(const auto& entry: transposedEntries) {
        int col = entry.first;
        int row = entry.second.first;
        int value = entry.second.second;

        while(current_col < col) {
            col_pointers.push_back(values.size());
            current_col++;
        }

        values.push_back(value);
        row_indices.push_back(row);
    }
    // The last element of col_pointers
    col_pointers.push_back(values.size());
    return true;
}

bool readMTXToTransposeCSR(const std::string& filename, 
                           std::vector<int>& values, 
                           std::vector<int>& row_indices, 
                           std::vector<int>& col_pointers) 
{
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
bool readMTXToCSRUndirected(const std::string& filename, std::vector<int>& values, std::vector<int>& column_indices, std::vector<int>& row_pointers) {
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

    std::multimap<int, std::pair<int, int>> entries;

    int row, col, val;
    for (int i = 0; i < numNonZero; ++i) {
        file >> row >> col >> val;
        row--; // Convert to 0-based indexing
        col--;

        entries.insert({row, {col, val}});
        if(col != row) // Avoid adding duplicates for self-loops
            entries.insert({col, {row, val}});
    }

    file.close();

    values.clear();
    column_indices.clear();
    row_pointers.clear();
    row_pointers.resize(numRows + 1, 0);

    int current_row = 0;
    int nnz = 0;
    for(const auto& entry: entries) {
        int row = entry.first;
        int col = entry.second.first;
        int value = entry.second.second;

        while (row > current_row) {
            row_pointers[current_row + 1] = nnz;
            current_row++;
        }

        values.push_back(value);
        column_indices.push_back(col);
        nnz++;
    }

    // Add the last row_pointer
    row_pointers[current_row + 1] = nnz;

    return true;
}

int main() {
    // std::vector<int> values, row_indices, col_pointers;
    // if(readMTXToTransposeCSR("graph.mtx", values, row_indices, col_pointers)) {
    //     std::cout << "Successfully read the MTX file and converted to Transpose CSR format!" << std::endl;
        
    //     std::cout << "Values Array: [";
    //     for(const auto& val : values) std::cout << val << ", ";
    //     std::cout << "\b\b]" << std::endl;

    //     std::cout << "Row Indices Array: [";
    //     for(const auto& idx : row_indices) std::cout << idx << ", ";
    //     std::cout << "\b\b]" << std::endl;
        
    //     std::cout << "Column Pointers Array: [";
    //     for(const auto& ptr : col_pointers) std::cout << ptr << ", ";
    //     std::cout << "\b\b]" << std::endl;
    // } else {
    //     std::cerr << "Failed to read the MTX file." << std::endl;
    // }

    // std::vector<int> values, row_indices, col_pointers;
    // if(readMTXToTransposeCSRUndirected("graph.mtx", values, row_indices, col_pointers)) {
    //     std::cout << "Successfully read the MTX file and converted to Transpose CSR format!\n";
    //     std::cout << "Values Array: "; for(const auto& val : values) std::cout << val << ' '; std::cout << '\n';
    //     std::cout << "Row Indices Array: "; for(const auto& idx : row_indices) std::cout << idx << ' '; std::cout << '\n';
    //     std::cout << "Column Pointers Array: "; for(const auto& ptr : col_pointers) std::cout << ptr << ' '; std::cout << '\n';
    // }


    std::vector<int> values, column_indices, row_pointers;
    if(readMTXToCSRUndirected("graph.mtx", values, column_indices, row_pointers)) {
        std::cout << "Successfully read the MTX file and converted to CSR format (Undirected)!\n";
        std::cout << "Values Array: "; for(const auto& val : values) std::cout << val << ' '; std::cout << '\n';
        std::cout << "Column Indices Array: "; for(const auto& idx : column_indices) std::cout << idx << ' '; std::cout << '\n';
        std::cout << "Row Pointers Array: "; for(const auto& ptr : row_pointers) std::cout << ptr << ' '; std::cout << '\n';
    }
    return 0;
}
