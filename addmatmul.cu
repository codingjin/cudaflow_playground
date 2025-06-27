#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/cuda/cudaflow.hpp>

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

// C = A + B
// E = C * D

// Kernel: add
__global__ void add(const int n, const float *x, const float *y, float *z) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

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
    const unsigned N = 1173;
    const unsigned N2 = N * N;
    const unsigned RANDOM_SEED = 137;
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> hA, hB, hC;
    hA.resize(N2);
    hB.resize(N2);
    hC.resize(N2, 0.0f);

    float* dA{nullptr};
    float* dB{nullptr};
    float* dC{nullptr};

    for (size_t i = 0; i < hA.size(); ++i) {
        hA[i] = 0.01;//dist(gen);
        hB[i] = 0.02;//dist(gen);
    }

    std::vector<float> std_C;
    std_C.resize(N2, 0.0f);

    for (size_t i = 0; i < std_C.size(); ++i) {
        std_C[i] = hA[i] + hB[i];
    }

    tf::Taskflow taskflow("AddMatmul");
    tf::Executor executor;

    // allocation for dA, dB, and dC (dC = dA + dB)
    auto allocate_dA = taskflow.emplace([&](){
      TF_CHECK_CUDA(cudaMalloc(&dA, hA.size()*sizeof(float)), "failed to allocate dA");
    }).name("allocate_dA");
    auto allocate_dB = taskflow.emplace([&](){
      TF_CHECK_CUDA(cudaMalloc(&dB, hB.size()*sizeof(float)), "failed to allocate dB");
    }).name("allocate_dB");
    auto allocate_dC = taskflow.emplace([&](){
      TF_CHECK_CUDA(cudaMalloc(&dC, hC.size()*sizeof(float)), "failed to allocate dC");
    }).name("allocate_dC");

    auto cuda_add_flow = taskflow.emplace([&](){
        tf::cudaGraph cg;
        auto copy_dA = cg.copy(dA, hA.data(), hA.size());
        auto copy_dB = cg.copy(dB, hB.data(), hB.size());

        auto copy_hC = cg.copy(hC.data(), dC, hC.size());

        dim3 grid((N2 + 1023) / 1024);
        dim3 block(1024);

        auto add_kernel = cg.kernel(grid, block, 0, add, N2, dA, dB, dC);
        add_kernel.succeed(copy_dA, copy_dB)
                  .precede(copy_hC);

        tf::cudaStream stream;
        tf::cudaGraphExec exec(cg);
        stream.run(exec).synchronize();
        //cg.dump(std::cout);
    }).name("cuda_add_flow");

    auto free_dA = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(dA), "failed to free dA");
    }).name("free_dA");
    auto free_dB = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(dB), "failed to free dB");
    }).name("free_dB");
    auto free_dC = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(dC), "failed to free dD");
    }).name("free_dC");

    /*
    cuda_add_flow.succeed(allocate_dA, allocate_dB, allocate_dC)
                 .precede(free_dA, free_dB, free_dC);
    executor.run(taskflow).wait();
    */
    
    // C = A + B
    // E = C * D
    std::vector<float> hD, hE, std_E;
    hD.resize(N2);
    hE.resize(N2, 0.0f);
    std_E.resize(N2, 0.0f);

    for (size_t i = 0; i < hD.size(); ++i) {
        hD[i] = 0.13;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float tmp = 0.0f;
            for (int k = 0; k < N; ++k) {
                tmp += std_C[i * N + k] * hD[k * N + j];
            }
            std_E[i * N + j] = tmp;
        }
    }

    float* dD{nullptr};
    float* dE{nullptr};

    auto allocate_dC2 = taskflow.emplace([&](){
      TF_CHECK_CUDA(cudaMalloc(&dC, hC.size()*sizeof(float)), "failed to allocate dC2");
    }).name("allocate_dC2");
    auto allocate_dD = taskflow.emplace([&](){
      TF_CHECK_CUDA(cudaMalloc(&dD, hD.size()*sizeof(float)), "failed to allocate dD");
    }).name("allocate_dD");
    auto allocate_dE = taskflow.emplace([&](){
      TF_CHECK_CUDA(cudaMalloc(&dE, hE.size()*sizeof(float)), "failed to allocate dE");
    }).name("allocate_dE");

    auto cuda_matmul_flow = taskflow.emplace([&](){
        tf::cudaGraph cg;
        auto copy_dC2 = cg.copy(dC, hC.data(), hC.size());
        auto copy_dD = cg.copy(dD, hD.data(), hD.size());

        auto copy_hE = cg.copy(hE.data(), dE, hE.size());

        dim3 grid((N + 31) / 32, (N + 31) / 32);
        dim3 block(32, 32);

        auto matmulkernel = cg.kernel(grid, block, 0, matmul, dC, dD, dE, N, N, N);
        matmulkernel.succeed(copy_dC2, copy_dD)
                    .precede(copy_hE);

        tf::cudaStream stream;
        tf::cudaGraphExec exec(cg);
        stream.run(exec).synchronize();
        //cg.dump(std::cout);
    }).name("cuda_matmul_flow");

    auto free_dC2 = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(dC), "failed to free dC2");
    }).name("free_dC2");
    auto free_dD = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(dD), "failed to free dD");
    }).name("free_dD");
    auto free_dE = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(dE), "failed to free dE");
    }).name("free_dE");

    auto wait = taskflow.emplace([](){}).name("wait");
    cuda_add_flow.succeed(allocate_dA, allocate_dB, allocate_dC)
                 .precede(free_dA, free_dB, free_dC);
    wait.succeed(free_dC)
        .precede(allocate_dC2, allocate_dD, allocate_dE);
    cuda_matmul_flow.succeed(allocate_dC2, allocate_dD, allocate_dE)
                    .precede(free_dC2, free_dD, free_dE);
    
    executor.run(taskflow).wait();

    /*
    std::cout << "Verifying results of add: C = A + B" << std::endl;
    float maxerr = 0.0f;
    for (size_t i = 0; i < std_C.size(); ++i) {
        maxerr = std::max(maxerr, std::abs(hC[i] - std_C[i]));
    }
    std::cout << "Max abs-error = " << maxerr << std::endl;
    */
    
    std::cout << "Verifying results of matmul: E = (A + B) * D" << std::endl;
    float maxerr = 0.0f;
    for (size_t i = 0; i < std_E.size(); ++i) {
        maxerr = std::max(maxerr, std::abs(hE[i] - std_E[i]));
    }
    std::cout << "Max abs-error = " << maxerr << std::endl;
    
    return 0;
}

