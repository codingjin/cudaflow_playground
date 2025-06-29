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

    tf::Taskflow taskflow("Matmul");
    tf::Executor executor;

    auto cudaFlow = taskflow.emplace([&](){
        tf::cudaGraph cg;
        auto copy_da = cg.copy(da, ha.data(), N2);
        auto copy_db = cg.copy(db, hb.data(), N2);
        auto copy_hc = cg.copy(hc.data(), dc, N2);

        auto matmulkernel = cg.kernel(grid, block, 0, matmul, da, db, dc, N, N, N);
        matmulkernel.succeed(copy_da, copy_db)
                    .precede(copy_hc);

        tf::cudaStream stream;
        tf::cudaGraphExec exec(cg);
        stream.run(exec).synchronize();
    }).name("cudaFlow");

    auto gbeg = std::chrono::steady_clock::now();
    for (int i = 0; i < IterationNum; ++i) {
        executor.run(taskflow).wait();
    }
    auto gend = std::chrono::steady_clock::now();

    std::cout << "CUDAFLOW matmul M=N=K=" << N << " IterationNum=" << IterationNum << std::endl 
        << "completed with " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(gend-gbeg).count()
        << " ms\n";

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    
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

