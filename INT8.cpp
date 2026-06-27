#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <immintrin.h>
#include <cmath>

using namespace std;

vector<int8_t> quantize_weights_avx2(float* W, int w, float& scale, int h) {
    int total_elements = w * h;
    vector<int8_t> W_int8(total_elements);

    float max_abs = 0.0f;
    for (int i = 0; i < total_elements; i++) { 
        max_abs = max(max_abs, abs(W[i]));
    }

    scale = max_abs / 127.0f;
    
    float inv_scale = 1.0f / scale; 
    __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);

    for (int i = 0; i <= total_elements - 8; i += 8) {
        
        __m256 float_vec = _mm256_loadu_ps(&W[i]);
        
        float_vec = _mm256_mul_ps(float_vec, inv_scale_vec);
        
        __m256i int32_vec = _mm256_cvtps_epi32(float_vec);
        __m128i int16_vec = _mm_packs_epi32(
            _mm256_castsi256_si128(int32_vec), 
            _mm256_extracti128_si256(int32_vec, 1) 
        );
        __m128i int8_vec = _mm_packs_epi16(int16_vec, int16_vec);
        
        _mm_storel_epi64((__m128i*)&W_int8[i], int8_vec);
    }
    
    return W_int8;
}

int main(){
    ifstream weight("multi_model_weights.bin", ios::binary);
    if(!weight.good()){
        cout << "Error opening weight file!" << endl;
        return -1;
    }

    vector<float> weights1(1024 * 512);
    vector<float> biases1(512);
    vector<float> weights2(512 * 128);
    vector<float> biases2(128);
    vector<float> weights3(128 * 10);
    vector<float> biases3(10);

    cout << "Reading 32-bit floats..." << endl;
    weight.read(reinterpret_cast<char*>(weights1.data()), weights1.size() * sizeof(float));
    weight.read(reinterpret_cast<char*>(biases1.data()), biases1.size() * sizeof(float));
    weight.read(reinterpret_cast<char*>(weights2.data()), weights2.size() * sizeof(float));
    weight.read(reinterpret_cast<char*>(biases2.data()), biases2.size() * sizeof(float));
    weight.read(reinterpret_cast<char*>(weights3.data()), weights3.size() * sizeof(float));
    weight.read(reinterpret_cast<char*>(biases3.data()), biases3.size() * sizeof(float));
    weight.close();

    float scale1;
    float scale2;
    float scale3;

    vector<int8_t> weight1 = quantize_weights_avx2(weights1.data(), 1024,scale1, 512);
    vector<int8_t> weight2 = quantize_weights_avx2(weights2.data(), 512, scale2, 128);
    vector<int8_t> weight3 = quantize_weights_avx2(weights3.data(), 128, scale3, 10);

     ofstream weight_out("quantized_weights.bin", ios::binary);
    
    cout << "Writing quantized_weights.bin..." << endl;
    
    weight_out.write(reinterpret_cast<char*>(&scale1), sizeof(float));                 
    weight_out.write(reinterpret_cast<char*>(weight1.data()), weight1.size());     
    weight_out.write(reinterpret_cast<char*>(biases1.data()), biases1.size() * 4);     
    
    weight_out.write(reinterpret_cast<char*>(&scale2), sizeof(float));
    weight_out.write(reinterpret_cast<char*>(weight2.data()), weight2.size());
    weight_out.write(reinterpret_cast<char*>(biases2.data()), biases2.size() * 4);
    
    weight_out.write(reinterpret_cast<char*>(&scale3), sizeof(float));
    weight_out.write(reinterpret_cast<char*>(weight3.data()), weight3.size());
    weight_out.write(reinterpret_cast<char*>(biases3.data()), biases3.size() * 4);
    
    weight_out.close();
    
    cout << "[SUCCESS] File quantized successfully!" << endl;

    return 0;
}