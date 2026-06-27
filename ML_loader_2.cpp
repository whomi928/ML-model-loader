#include <stdio.h>
#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <algorithm>
#include <chrono>

using namespace std;

void multiply_add_tiled(float* A, float* B, float* C, int M, int K, int N) {
    
    // A 64x64 block of floats is exactly 16 KB.
    // We load one block from A (16KB) and one from B (16KB) = 32KB total.
    // This perfectly saturates the CPU's L1 Data Cache.
    const int BLOCK_SIZE = 64;

    // =====================================================================
    // PHASE 1: THE OUTER 3 LOOPS (The Block Navigators)
    // These loops do no math. They simply move a "64x64 window" across the matrices.
    // =====================================================================
    for (int i_block = 0; i_block < M; i_block += BLOCK_SIZE) {
        for (int j_block = 0; j_block < N; j_block += BLOCK_SIZE) {
            for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
                
                // Calculate bounds to prevent Memory Faults on the edges 
                // (if matrix size isn't a perfect multiple of 64)
                int i_end = min(i_block + BLOCK_SIZE, M);
                int j_end = min(j_block + BLOCK_SIZE, N);
                int k_end = min(k_block + BLOCK_SIZE, K);

                // =====================================================================
                // PHASE 2: THE INNER 3 LOOPS (The Math Engine)
                // These loops ONLY run inside the current 64x64 window.
                // Because they are trapped in this small window, the L1 Cache is never evicted!
                // =====================================================================
                for (int i = i_block; i < i_end; i++) {
                    
                    // Step by 8 because AVX2 processes 8 floats per clock cycle
                    int j = j_block;
                    for (; j <= j_end - 8; j += 8) {
                        
                        // CRITICAL CHANGE: Because the 'k' loop is chopped into blocks, 
                        // we can't start at zero anymore! We must LOAD the existing 
                        // running total from Matrix C, add to it, and put it back.
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

int main(){

    cout << "heellloooo...." << endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    vector<float> A(1024 * 1024, 0.5f);
    vector<float> B(1024 * 1024, 0.5f);
    vector<float> C_tiled(1024 * 1024, 0.0f);

    multiply_add_tiled(A.data(), B.data(), C_tiled.data(), 1024, 1024, 1024);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    cout << "Engine Execution Time: " << duration.count() << " ms" << std::endl;

    return 0;
}