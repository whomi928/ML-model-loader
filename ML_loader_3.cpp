#include <stdio.h>
#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <fstream>

using namespace std;

void multiply_add_tiled(float* A, float* B, float* C, int M, int K, int N) {
    
    const int BLOCK_SIZE = 64;

    for (int i_block = 0; i_block < M; i_block += BLOCK_SIZE) {
        for (int j_block = 0; j_block < N; j_block += BLOCK_SIZE) {
            for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
                
                // Calculate bounds to prevent Memory Faults on the edges 
                // (if matrix size isn't a perfect multiple of 64)
                int i_end = min(i_block + BLOCK_SIZE, M);
                int j_end = min(j_block + BLOCK_SIZE, N);
                int k_end = min(k_block + BLOCK_SIZE, K);

                for (int i = i_block; i < i_end; i++) {
                    
                    // Step by 8 because AVX2 processes 8 floats per clock cycle
                    int j = j_block;
                    for (; j <= j_end - 8; j += 8) {
                        
                        __m256 sum_vec = _mm256_loadu_ps(&C[i * N + j]);
                        
                        for (int k = k_block; k < k_end; k++) {
                            // Broadcast 1 scalar from A
                            __m256 a_vec = _mm256_broadcast_ss(&A[i * K + k]);
                            
                            // Load 8 contiguous floats from B
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            
                            // Fused Multiply-Add (Accumulate into sum_vec)
                            sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        // STORE the accumulated results back into Matrix C
                        _mm256_storeu_ps(&C[i * N + j], sum_vec);
                    }
                    // NOTE: A scalar remainder loop would go here to handle the leftover columns 
                    // if (j_end - j_block) is not a perfect multiple of 8.
                    for(; j < j_end; j++){
                        for(int k = k_block; k < k_end; k++){
                            C[i * N + j] += A[i * K + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
class LinearLayer {
public: 
    int in_features;
    int out_features;

    vector<float> weights;  
    vector<float> biases;

    LinearLayer(int in, int out) : in_features(in), out_features(out) {
    weights.resize(in_features * out_features, 0.0f); 
    biases.resize(out_features, 0.0f);                
    }

    bool load_weights(ifstream& file) {
        if (!file.good()) {
            cerr << "[ERROR] File stream is invalid or corrupted!" << endl;
            return false;
        }

        // Pour the raw binary data directly into the std::vector RAM buffers
        file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));

        return true;
    }
    void forward(float *X, float* output, int batch_size) {
        multiply_add_tiled(X, weights.data(), output, batch_size, in_features, out_features);
        __m256 zero_vec = _mm256_setzero_ps();
        for (int i = 0; i < batch_size; i++) {
            int j = 0;
            for (; j <= out_features - 8; j += 8) {
                __m256 out_vec = _mm256_loadu_ps(&output[i * out_features + j]);
                __m256 bias_vec = _mm256_loadu_ps(&biases[j]);
                
                // Add Bias: Y = (X * W) + B
                out_vec = _mm256_add_ps(out_vec, bias_vec);
                
                // Apply ReLU: max(0, out_vec)
                out_vec = _mm256_max_ps(out_vec, zero_vec);
                
                // Store final activation
                _mm256_storeu_ps(&output[i * out_features + j], out_vec);
            }
            // Scalar remainder for Bias + ReLU
            for (; j < out_features; j++) {
                output[i * out_features + j] += biases[j];
                if (output[i * out_features + j] < 0.0f) output[i * out_features + j] = 0.0f;
            }
        }
    }
};

int main(){

    cout << "heellloooo...." << endl;
    
    //ok so i am doing this on my own 
    /*int BATCH_SIZE = 64; // Number of samples in a batch

    LinearLayer layer1(1024, 512);
    LinearLayer layer2(512, 128);
    LinearLayer layer3(128, 10);

    ifstream weight_file("multi_model_weights .bin", ios::binary); // will need to change the path address
    if (!weight_file) {
        cerr << "[ERROR] Could not open multi_model_weights.bin! Run Python script first." << endl;
        return -1;
    }

    layer1.load_weights(weight_file);
    layer2.load_weights(weight_file);
    layer3.load_weights(weight_file);
    weight_file.close();
    cout << "[SUCCESS] Weights mapped to AVX2 memory!" << endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    vector<float> input_data(BATCH_SIZE * 1024, 1.0f); // Dummy input data
    vector<float> hidden1(BATCH_SIZE * 512, 0.0f);     // Holds Layer 1 output
    vector<float> hidden2(BATCH_SIZE * 128, 0.0f);     // Holds Layer 2 output
    vector<float> final_output(BATCH_SIZE * 10, 0.0f); // Holds final predictions

    layer1.forward(input_data.data(), hidden1.data(), BATCH_SIZE);
    layer2.forward(hidden1.data(), hidden2.data(), BATCH_SIZE);
    layer3.forward(hidden2.data(), final_output.data(), BATCH_SIZE);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    cout << "Engine Execution Time: " << duration.count() << " ms" << endl;*/

    //this commented part is in alignment with the python script and also gives idea on how load data in linear layers.

    return 0;
}