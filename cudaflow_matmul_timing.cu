#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/cuda/cudaflow.hpp>

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>


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
    const unsigned N = 1137;
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
    

    tf::Taskflow taskflow("Matmul");
    tf::Executor executor;

    auto allocate_a = taskflow.emplace([&](){
      TF_CHECK_CUDA(cudaMalloc(&da, N2*sizeof(float)), "failed to allocate a");
    }).name("allocate_a");

    auto allocate_b = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&db, N2*sizeof(float)), "failed to allocate b");
    }).name("allocate_b");

    auto allocate_c = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&dc, N2*sizeof(float)), "failed to allocate c");
    }).name("allocate_c");

    auto cudaFlow = taskflow.emplace([&](){
        tf::cudaGraph cg;

        auto copy_da = cg.copy(da, ha.data(), N2);
        auto copy_db = cg.copy(db, hb.data(), N2);
        auto copy_hc = cg.copy(hc.data(), dc, N2);

        dim3 grid((N + 31) / 32, (N + 31) / 32);
        dim3 block(32, 32);

        auto matmulkernel = cg.kernel(grid, block, 0, matmul, da, db, dc, N, N, N);
        matmulkernel.succeed(copy_da, copy_db)
                    .precede(copy_hc);

        tf::cudaStream stream;
        tf::cudaGraphExec exec(cg);
        stream.run(exec).synchronize();
    }).name("cudaFlow");

    auto freea = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(da), "failed to free da"); 
    }).name("freea");

    auto freeb = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(db), "failed to free db"); 
    }).name("freeb");

    auto freec = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(dc), "failed to free dc"); 
    }).name("freec");

    cudaFlow.succeed(allocate_a, allocate_b, allocate_c)
            .precede(freea, freeb, freec);
    
    // Timing for 10 rounds
    for (int i = 0; i < 10; ++i) {
        auto gbeg = std::chrono::steady_clock::now();
        executor.run(taskflow).wait();
        auto gend = std::chrono::steady_clock::now();
        std::cout << "Round " << i << " completed with " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(gend-gbeg).count()
                << " ms\n";
    }
    
    std::cout << "Verifying results of Matmul C = A*B" << std::endl;
    float maxerr = 0.0f;
    for (size_t i = 0; i < N2; ++i) {
        maxerr = std::max(maxerr, std::abs(hc[i] - result[i]));
    }
    std::cout << "Max abs-error = " << maxerr << std::endl;
    
    return 0;
}

