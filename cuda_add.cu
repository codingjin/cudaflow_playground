#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/cuda/cudaflow.hpp>

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>


// Kernel: add
__global__ void add(const int n, const float a, const float *x, float *y) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a*x[idx] + y[idx];
    }
}

int main(int argc, const char** argv) {

    const unsigned N = 3237 * 3237;
    const float alpha = 0.737;
    const unsigned RANDOM_SEED = 137;
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> hx, hy;

    //* For verifying the results!
    std::vector<float> std_result;
    hx.resize(N);
    hy.resize(N);
    std_result.resize(N);
    for (size_t i = 0; i < N; ++i) {
        hx[i] = dist(gen);
        hy[i] = dist(gen);
        std_result[i] = alpha*hx[i] + hy[i];
    }
    //*/

    float* dx{nullptr};
    float* dy{nullptr};

    tf::Taskflow taskflow("Add");
    tf::Executor executor;

    auto allocate_x = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&dx, N*sizeof(float)), "failed to allocate dx");
    }).name("allocate_x");

    auto allocate_y = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&dy, N*sizeof(float)), "failed to allocate dy");
    }).name("allocate_y");

    
    // Create a cudaflow to run the add
    auto cudaFlow_add = taskflow.emplace([&](){
        tf::cudaGraph cg;

        auto copy_dx = cg.copy(dx, hx.data(), N);
        auto copy_dy = cg.copy(dy, hy.data(), N);
        auto copy_hy = cg.copy(hy.data(), dy, N);

        dim3 grid((N+1023)/1024);
        dim3 block(1024);

        auto add_kernel = cg.kernel(grid, block, 0, add, N, alpha, dx, dy);

        add_kernel.succeed(copy_dx, copy_dy)
                  .precede(copy_hy);

        tf::cudaStream stream;
        tf::cudaGraphExec exec(cg);
        stream.run(exec).synchronize();
        //cg.dump(std::cout);
    }).name("cudaFlow_add");

    auto free_dx = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(dx), "failed to free dx");
    }).name("free_dx");

    auto free_dy = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(dy), "failed to free dy");
    }).name("free_dy");

    cudaFlow_add.succeed(allocate_x, allocate_y)
                .precede(free_dx, free_dy);
    executor.run(taskflow).wait();

    //*
    std::cout << "Verifying results of Y = alpha*X + Y..." << std::endl;
    float maxerr = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        maxerr = std::max(maxerr, std::abs(hy[i] - std_result[i]));
    }
    std::cout << "Max abs-error = " << maxerr << std::endl;
    //*/
    
    return 0;
}




