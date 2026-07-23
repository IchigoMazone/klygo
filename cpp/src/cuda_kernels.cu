#include "klygo/cuda_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template<typename T, typename Op>
__global__ void binary_op_kernel(const T* a, const T* b, T* c, std::size_t n, Op op) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = op(a[idx], b[idx]);
    }
}

template<typename T>
struct AddOp { __device__ T operator()(T x, T y) const { return x + y; } };

template<typename T>
struct SubOp { __device__ T operator()(T x, T y) const { return x - y; } };

template<typename T>
struct MulOp { __device__ T operator()(T x, T y) const { return x * y; } };

template<typename T>
struct DivOp { __device__ T operator()(T x, T y) const { return x / y; } };

template<template<typename> class Op>
void launch_op(const void* a, const void* b, void* c, std::size_t n, DType dtype) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    switch (dtype) {
        case DType::Float32:
            binary_op_kernel<<<blocks, threads>>>(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(c), n, Op<float>());
            break;
        case DType::Float64:
            binary_op_kernel<<<blocks, threads>>>(static_cast<const double*>(a), static_cast<const double*>(b), static_cast<double*>(c), n, Op<double>());
            break;
        case DType::Int32:
            binary_op_kernel<<<blocks, threads>>>(static_cast<const int32_t*>(a), static_cast<const int32_t*>(b), static_cast<int32_t*>(c), n, Op<int32_t>());
            break;
        case DType::Int64:
            binary_op_kernel<<<blocks, threads>>>(static_cast<const int64_t*>(a), static_cast<const int64_t*>(b), static_cast<int64_t*>(c), n, Op<int64_t>());
            break;
        case DType::Bool:
            binary_op_kernel<<<blocks, threads>>>(static_cast<const bool*>(a), static_cast<const bool*>(b), static_cast<bool*>(c), n, Op<bool>());
            break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void launch_add_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) {
    launch_op<AddOp>(a, b, c, n, dtype);
}

void launch_sub_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) {
    launch_op<SubOp>(a, b, c, n, dtype);
}

void launch_mul_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) {
    launch_op<MulOp>(a, b, c, n, dtype);
}

void launch_div_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) {
    launch_op<DivOp>(a, b, c, n, dtype);
}
