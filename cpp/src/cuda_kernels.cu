#include "klygo/cuda_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * @brief CUDA Kernel tính toán element-wise giữa hai mảng dữ liệu (Binary Op Kernel).
 * Thực thi song song trên hàng nghìn luồng GPU thread.
 */
template<typename T, typename Op, typename OutT = T>
__global__ void binary_op_kernel(const T* a, const T* b, OutT* c, std::size_t n, Op op) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = op(a[idx], b[idx]);
    }
}

/**
 * @brief CUDA Kernel tính toán element-wise giữa một mảng và số thực (Scalar Op Kernel).
 */
template<typename T, typename Op, typename OutT = T>
__global__ void scalar_op_kernel(const T* a, double val_b, OutT* c, std::size_t n, Op op) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = op(a[idx], val_b);
    }
}

/**
 * @brief CUDA Kernel tính toán phép toán đơn biến trên một mảng (Unary Op Kernel).
 */
template<typename T, typename Op>
__global__ void unary_op_kernel(const T* a, T* c, std::size_t n, Op op) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = op(a[idx]);
    }
}

/**
 * @brief CUDA Kernel nhân ma trận 2D tối ưu hóa bằng Shared Memory Tiling (Shared Memory 16x16).
 * Giúp giảm số lần truy cập VRAM băng thông chậm và tăng tốc tính toán trên GPU.
 */
template<typename T>
__global__ void matmul_cuda_kernel(const T* A, const T* B, T* C, std::size_t M, std::size_t K, std::size_t N) {
    __shared__ T tileA[16][16];
    __shared__ T tileB[16][16];

    std::size_t row = blockIdx.y * 16 + threadIdx.y;
    std::size_t col = blockIdx.x * 16 + threadIdx.x;

    T sum = static_cast<T>(0);

    for (std::size_t t = 0; t < (K + 15) / 16; ++t) {
        if (row < M && (t * 16 + threadIdx.x) < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }

        if ((t * 16 + threadIdx.y) < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }

        __syncthreads();

        for (int k = 0; k < 16; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Struct Functors chạy trong môi trường thiết bị GPU (__device__)
template<typename T> struct AddOp { __device__ T operator()(T x, T y) const { return x + y; } };
template<typename T> struct SubOp { __device__ T operator()(T x, T y) const { return x - y; } };
template<typename T> struct MulOp { __device__ T operator()(T x, T y) const { return x * y; } };
template<typename T> struct DivOp { __device__ T operator()(T x, T y) const { return x / y; } };

// Comparison Functors trên GPU
template<typename T> struct EqOp { __device__ bool operator()(T x, T y) const { return x == y; } };
template<typename T> struct NeOp { __device__ bool operator()(T x, T y) const { return x != y; } };
template<typename T> struct LtOp { __device__ bool operator()(T x, T y) const { return x < y; } };
template<typename T> struct LeOp { __device__ bool operator()(T x, T y) const { return x <= y; } };
template<typename T> struct GtOp { __device__ bool operator()(T x, T y) const { return x > y; } };
template<typename T> struct GeOp { __device__ bool operator()(T x, T y) const { return x >= y; } };

// Scalar Functors trên GPU
template<typename T> struct AddScalarOp { __device__ T operator()(T x, double y) const { return static_cast<T>(x + y); } };
template<typename T> struct SubScalarOp { __device__ T operator()(T x, double y) const { return static_cast<T>(x - y); } };
template<typename T> struct MulScalarOp { __device__ T operator()(T x, double y) const { return static_cast<T>(x * y); } };
template<typename T> struct DivScalarOp { __device__ T operator()(T x, double y) const { return static_cast<T>(x / y); } };

// Scalar Comparison Functors trên GPU
template<typename T> struct EqScalarOp { __device__ bool operator()(T x, double y) const { return static_cast<double>(x) == y; } };
template<typename T> struct NeScalarOp { __device__ bool operator()(T x, double y) const { return static_cast<double>(x) != y; } };
template<typename T> struct LtScalarOp { __device__ bool operator()(T x, double y) const { return static_cast<double>(x) < y; } };
template<typename T> struct LeScalarOp { __device__ bool operator()(T x, double y) const { return static_cast<double>(x) <= y; } };
template<typename T> struct GtScalarOp { __device__ bool operator()(T x, double y) const { return static_cast<double>(x) > y; } };
template<typename T> struct GeScalarOp { __device__ bool operator()(T x, double y) const { return static_cast<double>(x) >= y; } };

// Unary Functors trên GPU
template<typename T> struct SqrtOp { __device__ T operator()(T x) const { return static_cast<T>(::sqrt(static_cast<double>(x))); } };
template<typename T> struct ExpOp  { __device__ T operator()(T x) const { return static_cast<T>(::exp(static_cast<double>(x))); } };
template<typename T> struct LogOp  { __device__ T operator()(T x) const { return static_cast<T>(::log(static_cast<double>(x))); } };
template<typename T> struct AbsOp  { __device__ T operator()(T x) const { return static_cast<T>(::fabs(static_cast<double>(x))); } };
template<typename T> struct NegOp  { __device__ T operator()(T x) const { return -x; } };

template<typename T>
struct PowScalarOp {
    double exp;
    __device__ T operator()(T x) const { return static_cast<T>(::pow(static_cast<double>(x), exp)); }
};

template<typename T>
struct ClampOp {
    double min_v;
    double max_v;
    __device__ T operator()(T x) const {
        double v = static_cast<double>(x);
        if (v < min_v) v = min_v;
        if (v > max_v) v = max_v;
        return static_cast<T>(v);
    }
};

// Hàm kích hoạt Binary Kernel trên GPU (Launch Helpers)
template<template<typename> class Op, typename OutT = void>
void launch_op(const void* a, const void* b, void* c, std::size_t n, DType dtype) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    switch (dtype) {
        case DType::Float32:
            binary_op_kernel<float, Op<float>, float><<<blocks, threads>>>(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(c), n, Op<float>());
            break;
        case DType::Float64:
            binary_op_kernel<double, Op<double>, double><<<blocks, threads>>>(static_cast<const double*>(a), static_cast<const double*>(b), static_cast<double*>(c), n, Op<double>());
            break;
        case DType::Int32:
            binary_op_kernel<int32_t, Op<int32_t>, int32_t><<<blocks, threads>>>(static_cast<const int32_t*>(a), static_cast<const int32_t*>(b), static_cast<int32_t*>(c), n, Op<int32_t>());
            break;
        case DType::Int64:
            binary_op_kernel<int64_t, Op<int64_t>, int64_t><<<blocks, threads>>>(static_cast<const int64_t*>(a), static_cast<const int64_t*>(b), static_cast<int64_t*>(c), n, Op<int64_t>());
            break;
        case DType::Bool:
            binary_op_kernel<bool, Op<bool>, bool><<<blocks, threads>>>(static_cast<const bool*>(a), static_cast<const bool*>(b), static_cast<bool*>(c), n, Op<bool>());
            break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

// Hàm kích hoạt Comparison Kernel trên GPU
template<template<typename> class Op>
void launch_comp_op(const void* a, const void* b, void* c, std::size_t n, DType dtype) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    switch (dtype) {
        case DType::Float32:
            binary_op_kernel<float, Op<float>, bool><<<blocks, threads>>>(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<bool*>(c), n, Op<float>());
            break;
        case DType::Float64:
            binary_op_kernel<double, Op<double>, bool><<<blocks, threads>>>(static_cast<const double*>(a), static_cast<const double*>(b), static_cast<bool*>(c), n, Op<double>());
            break;
        case DType::Int32:
            binary_op_kernel<int32_t, Op<int32_t>, bool><<<blocks, threads>>>(static_cast<const int32_t*>(a), static_cast<const int32_t*>(b), static_cast<bool*>(c), n, Op<int32_t>());
            break;
        case DType::Int64:
            binary_op_kernel<int64_t, Op<int64_t>, bool><<<blocks, threads>>>(static_cast<const int64_t*>(a), static_cast<const int64_t*>(b), static_cast<bool*>(c), n, Op<int64_t>());
            break;
        case DType::Bool:
            binary_op_kernel<bool, Op<bool>, bool><<<blocks, threads>>>(static_cast<const bool*>(a), static_cast<const bool*>(b), static_cast<bool*>(c), n, Op<bool>());
            break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

// Hàm kích hoạt Scalar Kernel trên GPU
template<template<typename> class Op>
void launch_scalar_op(const void* a, double val_b, void* c, std::size_t n, DType dtype) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    switch (dtype) {
        case DType::Float32:
            scalar_op_kernel<float, Op<float>, float><<<blocks, threads>>>(static_cast<const float*>(a), val_b, static_cast<float*>(c), n, Op<float>());
            break;
        case DType::Float64:
            scalar_op_kernel<double, Op<double>, double><<<blocks, threads>>>(static_cast<const double*>(a), val_b, static_cast<double*>(c), n, Op<double>());
            break;
        case DType::Int32:
            scalar_op_kernel<int32_t, Op<int32_t>, int32_t><<<blocks, threads>>>(static_cast<const int32_t*>(a), val_b, static_cast<int32_t*>(c), n, Op<int32_t>());
            break;
        case DType::Int64:
            scalar_op_kernel<int64_t, Op<int64_t>, int64_t><<<blocks, threads>>>(static_cast<const int64_t*>(a), val_b, static_cast<int64_t*>(c), n, Op<int64_t>());
            break;
        case DType::Bool:
            scalar_op_kernel<bool, Op<bool>, bool><<<blocks, threads>>>(static_cast<const bool*>(a), val_b, static_cast<bool*>(c), n, Op<bool>());
            break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

// Hàm kích hoạt Scalar Comparison Kernel trên GPU
template<template<typename> class Op>
void launch_scalar_comp_op(const void* a, double val_b, void* c, std::size_t n, DType dtype) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    switch (dtype) {
        case DType::Float32:
            scalar_op_kernel<float, Op<float>, bool><<<blocks, threads>>>(static_cast<const float*>(a), val_b, static_cast<bool*>(c), n, Op<float>());
            break;
        case DType::Float64:
            scalar_op_kernel<double, Op<double>, bool><<<blocks, threads>>>(static_cast<const double*>(a), val_b, static_cast<bool*>(c), n, Op<double>());
            break;
        case DType::Int32:
            scalar_op_kernel<int32_t, Op<int32_t>, bool><<<blocks, threads>>>(static_cast<const int32_t*>(a), val_b, static_cast<bool*>(c), n, Op<int32_t>());
            break;
        case DType::Int64:
            scalar_op_kernel<int64_t, Op<int64_t>, bool><<<blocks, threads>>>(static_cast<const int64_t*>(a), val_b, static_cast<bool*>(c), n, Op<int64_t>());
            break;
        case DType::Bool:
            scalar_op_kernel<bool, Op<bool>, bool><<<blocks, threads>>>(static_cast<const bool*>(a), val_b, static_cast<bool*>(c), n, Op<bool>());
            break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

// Hàm kích hoạt Unary Kernel trên GPU
template<template<typename> class Op>
void launch_unary_op(const void* a, void* c, std::size_t n, DType dtype) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    switch (dtype) {
        case DType::Float32:
            unary_op_kernel<<<blocks, threads>>>(static_cast<const float*>(a), static_cast<float*>(c), n, Op<float>());
            break;
        case DType::Float64:
            unary_op_kernel<<<blocks, threads>>>(static_cast<const double*>(a), static_cast<double*>(c), n, Op<double>());
            break;
        case DType::Int32:
            unary_op_kernel<<<blocks, threads>>>(static_cast<const int32_t*>(a), static_cast<int32_t*>(c), n, Op<int32_t>());
            break;
        case DType::Int64:
            unary_op_kernel<<<blocks, threads>>>(static_cast<const int64_t*>(a), static_cast<int64_t*>(c), n, Op<int64_t>());
            break;
        case DType::Bool:
            unary_op_kernel<<<blocks, threads>>>(static_cast<const bool*>(a), static_cast<bool*>(c), n, Op<bool>());
            break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

// === Triển khai Binary Launchers ===
void launch_add_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) { launch_op<AddOp>(a, b, c, n, dtype); }
void launch_sub_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) { launch_op<SubOp>(a, b, c, n, dtype); }
void launch_mul_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) { launch_op<MulOp>(a, b, c, n, dtype); }
void launch_div_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) { launch_op<DivOp>(a, b, c, n, dtype); }

// === Triển khai Comparison Launchers ===
void launch_eq_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) { launch_comp_op<EqOp>(a, b, c, n, dtype); }
void launch_ne_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) { launch_comp_op<NeOp>(a, b, c, n, dtype); }
void launch_lt_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) { launch_comp_op<LtOp>(a, b, c, n, dtype); }
void launch_le_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) { launch_comp_op<LeOp>(a, b, c, n, dtype); }
void launch_gt_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) { launch_comp_op<GtOp>(a, b, c, n, dtype); }
void launch_ge_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype) { launch_comp_op<GeOp>(a, b, c, n, dtype); }

// === Triển khai Scalar Launchers ===
void launch_add_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype) { launch_scalar_op<AddScalarOp>(a, b, c, n, dtype); }
void launch_sub_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype) { launch_scalar_op<SubScalarOp>(a, b, c, n, dtype); }
void launch_mul_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype) { launch_scalar_op<MulScalarOp>(a, b, c, n, dtype); }
void launch_div_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype) { launch_scalar_op<DivScalarOp>(a, b, c, n, dtype); }

// === Triển khai Scalar Comparison Launchers ===
void launch_eq_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype) { launch_scalar_comp_op<EqScalarOp>(a, b, c, n, dtype); }
void launch_ne_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype) { launch_scalar_comp_op<NeScalarOp>(a, b, c, n, dtype); }
void launch_lt_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype) { launch_scalar_comp_op<LtScalarOp>(a, b, c, n, dtype); }
void launch_le_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype) { launch_scalar_comp_op<LeScalarOp>(a, b, c, n, dtype); }
void launch_gt_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype) { launch_scalar_comp_op<GtScalarOp>(a, b, c, n, dtype); }
void launch_ge_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype) { launch_scalar_comp_op<GeScalarOp>(a, b, c, n, dtype); }

// === Triển khai Unary Launchers ===
void launch_pow_scalar_cuda(const void* a, double exponent, void* c, std::size_t n, DType dtype) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    switch (dtype) {
        case DType::Float32: unary_op_kernel<<<blocks, threads>>>(static_cast<const float*>(a), static_cast<float*>(c), n, PowScalarOp<float>{exponent}); break;
        case DType::Float64: unary_op_kernel<<<blocks, threads>>>(static_cast<const double*>(a), static_cast<double*>(c), n, PowScalarOp<double>{exponent}); break;
        case DType::Int32:   unary_op_kernel<<<blocks, threads>>>(static_cast<const int32_t*>(a), static_cast<int32_t*>(c), n, PowScalarOp<int32_t>{exponent}); break;
        case DType::Int64:   unary_op_kernel<<<blocks, threads>>>(static_cast<const int64_t*>(a), static_cast<int64_t*>(c), n, PowScalarOp<int64_t>{exponent}); break;
        case DType::Bool:    unary_op_kernel<<<blocks, threads>>>(static_cast<const bool*>(a), static_cast<bool*>(c), n, PowScalarOp<bool>{exponent}); break;
    }
}
void launch_sqrt_cuda(const void* a, void* c, std::size_t n, DType dtype) { launch_unary_op<SqrtOp>(a, c, n, dtype); }
void launch_exp_cuda(const void* a, void* c, std::size_t n, DType dtype)  { launch_unary_op<ExpOp>(a, c, n, dtype); }
void launch_log_cuda(const void* a, void* c, std::size_t n, DType dtype)  { launch_unary_op<LogOp>(a, c, n, dtype); }
void launch_abs_cuda(const void* a, void* c, std::size_t n, DType dtype)  { launch_unary_op<AbsOp>(a, c, n, dtype); }
void launch_neg_cuda(const void* a, void* c, std::size_t n, DType dtype)  { launch_unary_op<NegOp>(a, c, n, dtype); }

void launch_clamp_cuda(const void* a, double min_val, double max_val, void* c, std::size_t n, DType dtype) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    switch (dtype) {
        case DType::Float32: unary_op_kernel<<<blocks, threads>>>(static_cast<const float*>(a), static_cast<float*>(c), n, ClampOp<float>{min_val, max_val}); break;
        case DType::Float64: unary_op_kernel<<<blocks, threads>>>(static_cast<const double*>(a), static_cast<double*>(c), n, ClampOp<double>{min_val, max_val}); break;
        case DType::Int32:   unary_op_kernel<<<blocks, threads>>>(static_cast<const int32_t*>(a), static_cast<int32_t*>(c), n, ClampOp<int32_t>{min_val, max_val}); break;
        case DType::Int64:   unary_op_kernel<<<blocks, threads>>>(static_cast<const int64_t*>(a), static_cast<int64_t*>(c), n, ClampOp<int64_t>{min_val, max_val}); break;
        case DType::Bool:    unary_op_kernel<<<blocks, threads>>>(static_cast<const bool*>(a), static_cast<bool*>(c), n, ClampOp<bool>{min_val, max_val}); break;
    }
}

// === Triển khai Tiled Matmul GPU Launcher ===
void launch_matmul_cuda(const void* a, const void* b, void* c, std::size_t M, std::size_t K, std::size_t N, DType dtype) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(static_cast<unsigned int>((N + 15) / 16), static_cast<unsigned int>((M + 15) / 16));
    switch (dtype) {
        case DType::Float32:
            matmul_cuda_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(c), M, K, N);
            break;
        case DType::Float64:
            matmul_cuda_kernel<double><<<blocksPerGrid, threadsPerBlock>>>(static_cast<const double*>(a), static_cast<const double*>(b), static_cast<double*>(c), M, K, N);
            break;
        case DType::Int32:
            matmul_cuda_kernel<int32_t><<<blocksPerGrid, threadsPerBlock>>>(static_cast<const int32_t*>(a), static_cast<const int32_t*>(b), static_cast<int32_t*>(c), M, K, N);
            break;
        case DType::Int64:
            matmul_cuda_kernel<int64_t><<<blocksPerGrid, threadsPerBlock>>>(static_cast<const int64_t*>(a), static_cast<const int64_t*>(b), static_cast<int64_t*>(c), M, K, N);
            break;
        case DType::Bool:
            matmul_cuda_kernel<bool><<<blocksPerGrid, threadsPerBlock>>>(static_cast<const bool*>(a), static_cast<const bool*>(b), static_cast<bool*>(c), M, K, N);
            break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA matmul kernel launch failed: ") + cudaGetErrorString(err));
    }
}
