// Core CUDA headers
#include <cuda_runtime.h>          // Mandatory
#include <device_launch_parameters.h> // Recommended for kernel variables

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>


// Matmul kernel
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N && row < M) {
        float tmp = 0.0f;
        for (int i = 0; i < N; ++i) {
            tmp += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = tmp;
    }
}


int main() {
    const unsigned N = 2177;
    const unsigned N2 = N*N;
    const unsigned RANDOM_SEED = 137;
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> ha, hb, hc;
    float* da{nullptr};
    float* db{nullptr};
    float* dc{nullptr};

    
    std::vector<float> result;
    ha.resize(N2);
    hb.resize(N2);
    hc.resize(N2, 0.0f);
    result.resize(N2);

    for (size_t i = 0; i < N2; ++i) {
        ha[i] = dist(gen);
        hb[i] = dist(gen);
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float tmp = 0.0f;
            for (int k = 0; k < N; ++k)
                tmp += ha[i * N + k] * hb[k * N + j];
            result[i * N + j] = tmp;
        }
    }
    
    dim3 grid((N + 31) / 32, (N + 31) / 32);
    dim3 block(32, 32);

    // Timing for 10 rounds
    for (int i = 0; i < 10; ++i) {
        auto gbeg = std::chrono::steady_clock::now();
        cudaMalloc(&da, N2*sizeof(float));
        cudaMalloc(&db, N2*sizeof(float));
        cudaMalloc(&dc, N2*sizeof(float));

        cudaMemcpy(da, &ha[0], ha.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, &hb[0], hb.size() * sizeof(float), cudaMemcpyHostToDevice);

        matmul<<<grid, block>>>(da, db, dc, N, N, N);
        cudaMemcpy(&hc[0], dc, hc.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
        auto gend = std::chrono::steady_clock::now();

        std::cout << "Round " << i << " completed with " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(gend-gbeg).count()
                << " ms\n";
    }

    
    
    std::cout << "Verifying results of Matmul C = A*B" << std::endl;
    std::cout << "Problem Size: M = N = K = " << N << std::endl;
    float maxerr = 0.0f;
    for (size_t i = 0; i < N2; ++i) {
        maxerr = std::max(maxerr, std::abs(hc[i] - result[i]));
    }
    std::cout << "Max abs-error = " << maxerr << std::endl;
    
    return 0;
}

