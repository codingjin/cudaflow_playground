// Core CUDA headers
#include <cuda_runtime.h>          // Mandatory
#include <device_launch_parameters.h> // Recommended for kernel variables

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <stdexcept> // For std::invalid_argument, std::out_of_range


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


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <integer>(>=1)" << std::endl;
        return 1; // Return error code
    }
    int IterationNum = 0;
    std::string arg = argv[1]; // Get the argument string
    try {
        size_t pos;
        int num = std::stoi(arg, &pos); // Convert to integer

        // Ensure entire string was processed (no extra characters)
        if (pos != arg.length()) {
            std::cerr << "Error: Argument must be a single integer." << std::endl;
            return 1;
        }

        if (num < 1) {
            std::cerr << "Error: Argument(integer) should be >= 1." << std::endl;
            return 1;
        }

        IterationNum = num;
    }
    catch (const std::invalid_argument&) {
        std::cerr << "Error: '" << arg << "' is not a valid integer." << std::endl;
        return 1;
    }
    catch (const std::out_of_range&) {
        std::cerr << "Error: '" << arg << "' is out of int range." << std::endl;
        return 1;
    }

    const unsigned N = 4096;
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

    dim3 grid((N + 31) / 32, (N + 31) / 32);
    dim3 block(32, 32);

    cudaMalloc(&da, N2*sizeof(float));
    cudaMalloc(&db, N2*sizeof(float));
    cudaMalloc(&dc, N2*sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    auto gbeg = std::chrono::steady_clock::now();
    for (int i = 0; i < IterationNum; ++i) {
        cudaMemcpyAsync(da, &ha[0], N2*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(db, &hb[0], N2*sizeof(float), cudaMemcpyHostToDevice, stream);
        matmul<<<grid, block, 0, stream>>>(da, db, dc, N, N, N);
        cudaMemcpyAsync(&hc[0], dc, N2*sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    auto gend = std::chrono::steady_clock::now();
    
    std::cout << "CUDA matmul M=N=K=" << N << " IterationNum=" << IterationNum << std::endl 
        << "completed with " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(gend-gbeg).count()
        << " ms\n";
    
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaStreamDestroy(stream);
    /*
    std::cout << "Verifying results of Matmul C = A*B" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float tmp = 0.0f;
            for (int k = 0; k < N; ++k)
                tmp += ha[i * N + k] * hb[k * N + j];
            result[i * N + j] = tmp;
        }
    }

    float maxerr = 0.0f;
    for (size_t i = 0; i < N2; ++i) {
        maxerr = std::max(maxerr, std::abs(hc[i] - result[i]));
    }
    std::cout << "Max abs-error = " << maxerr << std::endl;
    */
    return 0;
}

