#include <stdio.h>
#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <algorithm>
#include <chrono>

using namespace std;

//load data here, maybe, as we can use it anywhere.

void multiply_add(float* a, float* b, float* final, int row_1, int row_2, int col_1, int col_2){ // nXm , mXo 1st col. number is same as the 2nd row number
    __m256 zero = _mm256_setzero_ps();
    __m256 value = zero;
    for(int i = 0; i < row_1; i++){
        int j = 0;
        for(; j < col_2; j+=8){
            for(int k = 0; k < row_2; k++){
                //sum += a[i * col_1 + k] * b[k * col_2 + j];
                __m256 a_vec = _mm256_broadcast_ss(&a[i * col_1 + k]);
                __m256 b_vec = _mm256_loadu_ps(&b[k * col_2 + j]);

                value = _mm256_fmadd_ps(a_vec, b_vec, value); // Fused multiply-add: value = a_vec * b_vec + zero
                
            }
            _mm256_storeu_ps(&final[i * col_2 + j], value);
            value = zero;
        }
        for(; j < col_2; j++){
            for(int k = 0; k < row_2; k++){
                final[i * col_2 + j] +=  a[i * col_1 + k] * b[k * col_2 + j];
            }
        }
    }
}

int main(){

    cout << "heellloooo...." << endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    vector<float> A(1024 * 1024, 0.5f);
    vector<float> B(1024 * 1024, 0.5f);
    vector<float> C_tiled(1024 * 1024, 0.0f);

    multiply_add(A.data(), B.data(), C_tiled.data(), 1024, 1024, 1024, 1024);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    cout << "Engine Execution Time: " << duration.count() << " ms" << std::endl;

    return 0;
}