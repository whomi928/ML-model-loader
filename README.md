# ML Inference Engine — Bare-Metal CPU Inference in C++

A from-scratch neural network inference engine written in C++ that runs trained models **directly on CPU with zero ML framework dependencies**. Implements cache-tiled GEMM, AVX2/SIMD acceleration, and INT8 quantization — the same techniques used in [ggml](https://github.com/ggerganov/ggml) and [llama.cpp](https://github.com/ggerganov/llama.cpp).

---

## Performance

| Model Architecture | Precision | Inference Time |
|---|---|---|
| 10→512→512→128→10 (Linear NN) | FP32 | ~8ms |
| 10→512→512→128→10 (Linear NN) | INT8 (quantized) | ~8ms |

Benchmarked using `std::chrono` on CPU. No GPU. No PyTorch. No ONNX Runtime.

---

## What This Does

Most inference engines rely on PyTorch or TensorFlow, which carry gigabytes of overhead and abstract away all the interesting hardware behavior. This engine:

- **Loads binary weight files** (`.bin` format) directly into memory
- **Runs forward passes** through fully-connected (linear) neural networks
- **Implements cache tiling** to keep matrix data in L1 cache during GEMM operations
- **Uses AVX2 SIMD intrinsics** (`_mm256_fmadd_ps`) to process 8 floats per clock cycle
- **Quantizes FP32 weights to INT8** to halve memory bandwidth requirements and improve throughput

---

## Why It Matters

A naive matrix multiply is O(N³) and immediately thrashes the L1 cache on any large model. This engine solves that with two techniques:

**1. Cache Tiling:** Matrices are broken into small blocks (~64×64) sized to fit inside the L1 cache. This eliminates cache misses during the inner multiply loop — the single biggest bottleneck in CPU-side inference.

**2. AVX2 SIMD:** Instead of multiplying one float at a time, `_mm256_fmadd_ps` performs a fused multiply-add on 8 floats simultaneously in a single instruction. On a modern CPU, this alone gives a theoretical 8× speedup over scalar code on the compute-bound inner loop.

**3. INT8 Quantization:** FP32 weights are compressed to INT8, reducing model memory footprint by 4×. Inference runs using `_mm256_madd_epi16`, operating on integer vectors. Reduces RAM bandwidth pressure significantly on large weight matrices.

---

## Architecture

```
[ Python Training (Colab) ]
          |
          | exports
          v
[ multi_model_weights.bin ]   (FP32 binary weight dump)
[ quantized_weights.bin   ]   (INT8 quantized weights)
          |
          | loaded by
          v
[ ML_loader_3.cpp ]
  ├── Weight loader (binary deserialization)
  ├── GEMM kernel (cache-tiled, AVX2)
  ├── INT8 quantization runtime
  └── Chrono benchmarking
```

---

## Getting Started

### Prerequisites

- GCC or Clang with AVX2 support (`-mavx2 -mfma` flags)
- x86-64 CPU (Intel Haswell / AMD Ryzen or newer)
- VS Code (optional, for the provided JSON config)

### Build & Run (FP32)

```bash
# Clone the repo
git clone https://github.com/whomi928/ML-model-loader
cd ML-model-loader

# Compile with AVX2 enabled
g++ -O3 -mavx2 -mfma -o ML_loader_3 ML_loader_3.cpp

# Run
./ML_loader_3
```

Place `multi_model_weights.bin` in the same directory as the binary before running.

### Build & Run (INT8 Quantized)

In `ML_loader_3.cpp`, update the weight file path to point to `quantized_weights.bin`, then rebuild and run.

### Generating Weights (Python / Colab)

1. Open and run the included Colab notebook
2. It trains a small linear network and exports `multi_model_weights.bin`
3. Download the file and place it next to `ML_loader_3`

### VS Code SIMD Support

Add the provided `c_cpp_properties.json` to your `.vscode/` folder to enable IntelliSense for AVX2 intrinsics.

---

## Technical Details

### Weight File Format

Weights are stored as raw binary dumps of FP32 arrays, layer by layer. The loader reads layer dimensions from a header, then `memcpy`s directly into aligned buffers for SIMD-safe access.

### SIMD Alignment

All weight buffers are allocated with `_mm_malloc` (32-byte aligned) to avoid unaligned load penalties with AVX2.

### Quantization Scheme

FP32 weights are quantized using symmetric per-tensor quantization:

```
scale = max(|W|) / 127
W_int8 = round(W / scale)
```

Dequantization is applied after accumulation to recover FP32 outputs.

---

## Roadmap

- [ ] Convolutional layer support (2D GEMM tiling)
- [ ] Multi-threading via `std::thread` or OpenMP across tiles
- [ ] ONNX model import
- [ ] ARM NEON port (for mobile/embedded targets)

---

## Related Projects

- [ggml](https://github.com/ggerganov/ggml) — the inspiration for this approach
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — production CPU inference using the same techniques
- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) — reference for high-performance GEMM

---

## Author

**Shaurya Aditya** — B.Tech ECE, IIT BHU  
[LinkedIn](https://www.linkedin.com/in/shaurya-aditya-0563a0377) · [GitHub](https://github.com/whomi928)
