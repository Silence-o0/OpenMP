#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <chrono>

using namespace std;

pair<vector<double>, double> multiply_matrix_by_vector_and_max(const vector<vector<double>>& matrix, const vector<double>& vec) {
    vector<double> result(matrix.size(), 0.0);
    double max_elem = 0.0;

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
        if (result[i] > max_elem) {
            max_elem = result[i];
        }
    }
    return {result, max_elem};
}

pair<vector<double>, double> normalize_and_compare(const vector<double>& new_vector, const vector<double>& old_vector, double max_elem) {
    vector<double> normalize_vector(new_vector.size(), 0.0);
    double div_sum = 0.0;

    for (size_t i = 0; i < new_vector.size(); ++i) {
        normalize_vector[i] = new_vector[i] / max_elem;
        div_sum += pow(normalize_vector[i] - old_vector[i], 2);
    }

    return {normalize_vector, div_sum};
}

pair<vector<vector<double>>, size_t> read_input(const string& input_file_name) {
    ifstream input_file(input_file_name);
    if (!input_file.is_open()) {
        throw runtime_error("Error: Unable to open input file.");
    }

    size_t n;
    input_file >> n;

    vector<vector<double>> matrix(n, vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            input_file >> matrix[i][j];
        }
    }

    input_file.close();
    return {matrix, n};
}

int sequential(string input_file_name) {
    const double epsilon = 0.001;

    try {
        auto [matrix, n] = read_input(input_file_name);

        vector<double> eigenvalue_vector(n, 1.0);
        double div_sum = 1.0;

        auto start_time = chrono::high_resolution_clock::now();

        while (div_sum >= epsilon) {
            auto [multiply_result, max_val] = multiply_matrix_by_vector_and_max(matrix, eigenvalue_vector);
            auto [norm_vector, new_div_sum] = normalize_and_compare(multiply_result, eigenvalue_vector, max_val);
            eigenvalue_vector = norm_vector;
            div_sum = new_div_sum;
        }

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;
        double total_time = elapsed.count();

        cout << "Eigenvalue vector: ";
        for (double val : eigenvalue_vector) {
            cout << val << " ";
        }
        cout << "\n";
        cout << "Time: " << total_time << " seconds.\n";

    } catch (const exception& e) {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}