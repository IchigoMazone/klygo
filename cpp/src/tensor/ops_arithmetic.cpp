#include <cstring>
#include <functional>
#include <stdexcept>
#include "klygo/tensor.h"
#include "klygo/tensor_utils.h"

using namespace std;
using namespace klygo_internal;

// === Phép cộng hai Tensor (Add) ===
Tensor Tensor::add(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::plus<double>(), launch_add_cuda);
}

// === Phép trừ hai Tensor (Sub) ===
Tensor Tensor::sub(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::minus<double>(), launch_sub_cuda);
}

// === Phép nhân hai Tensor (Mul) ===
Tensor Tensor::mul(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::multiplies<double>(), launch_mul_cuda);
}

// === Phép chia hai Tensor (Div) ===
Tensor Tensor::div(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::divides<double>(), launch_div_cuda);
}

// === Phép cộng Tensor với Số thực (Add Scalar) ===
Tensor Tensor::add(double other) const {
    return scalar_op_dispatch_cuda(*this, other, std::plus<double>(), launch_add_scalar_cuda);
}

// === Phép trừ Tensor với Số thực (Sub Scalar) ===
Tensor Tensor::sub(double other) const {
    return scalar_op_dispatch_cuda(*this, other, std::minus<double>(), launch_sub_scalar_cuda);
}

// === Phép nhân Tensor với Số thực (Mul Scalar) ===
Tensor Tensor::mul(double other) const {
    return scalar_op_dispatch_cuda(*this, other, std::multiplies<double>(), launch_mul_scalar_cuda);
}

// === Phép chia Tensor với Số thực (Div Scalar) ===
Tensor Tensor::div(double other) const {
    return scalar_op_dispatch_cuda(*this, other, std::divides<double>(), launch_div_scalar_cuda);
}

// === Phép cộng trực tiếp trên Tensor hiện tại (In-place Add_) ===
Tensor& Tensor::add_(const Tensor& other) {
    Tensor res = add(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}

Tensor& Tensor::add_(double other) {
    Tensor res = add(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}

// === Phép trừ trực tiếp trên Tensor hiện tại (In-place Sub_) ===
Tensor& Tensor::sub_(const Tensor& other) {
    Tensor res = sub(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}

Tensor& Tensor::sub_(double other) {
    Tensor res = sub(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}

// === Phép nhân trực tiếp trên Tensor hiện tại (In-place Mul_) ===
Tensor& Tensor::mul_(const Tensor& other) {
    Tensor res = mul(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}

Tensor& Tensor::mul_(double other) {
    Tensor res = mul(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}

// === Phép chia trực tiếp trên Tensor hiện tại (In-place Div_) ===
Tensor& Tensor::div_(const Tensor& other) {
    Tensor res = div(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}

Tensor& Tensor::div_(double other) {
    Tensor res = div(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}

// === Gán toàn bộ giá trị phần tử bằng value (Fill_) ===
Tensor& Tensor::fill_(double value) {
    Tensor f = full(shape(), value, dtype());
    std::memcpy(data(), f.data(), numel() * dtype_size(dtype()));
    return *this;
}

// === Gán toàn bộ giá trị phần tử bằng 0 (Zero_) ===
Tensor& Tensor::zero_() {
    return fill_(0.0);
}

/**
 * @brief Vòng lặp nhân ma trận CPU tối ưu theo thứ tự r-k-c (Cache-friendly loop).
 * Vòng lặp bên trong cùng tăng chỉ số c trên mảng liên tiếp, tối ưu L1/L2 Cache và vector SIMD.
 */
template<typename TA, typename TB, typename TR>
void matmul_typed(const TA* a_ptr, const TB* b_ptr, TR* res_ptr, std::size_t M, std::size_t K, std::size_t N) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < static_cast<int64_t>(M); ++r) {
        for (std::size_t k = 0; k < K; ++k) {
            TR val_a = static_cast<TR>(a_ptr[r * K + k]);
            TR* res_row = res_ptr + r * N;
            const TB* b_row = b_ptr + k * N;
            for (std::size_t c = 0; c < N; ++c) {
                res_row[c] += val_a * static_cast<TR>(b_row[c]);
            }
        }
    }
}

// === Phép Nhân Ma trận 2D (Matrix Multiplication @ / matmul) ===
Tensor Tensor::matmul(const Tensor& other) const {
    if (dim() != 2 || other.dim() != 2) {
        throw std::runtime_error("matmul expects 2D tensors");
    }
    if (size(1) != other.size(0)) {
        throw std::runtime_error("Matrix inner dimensions must match");
    }
    
    std::size_t M = size(0);
    std::size_t K = size(1);
    std::size_t N = other.size(1);
    
    DType res_dtype = promote_types(dtype(), other.dtype());
    
    // Nếu trên thiết bị GPU, gọi tới Tiled 2D CUDA Matmul Kernel
    if (device().type() == DeviceType::CUDA) {
#ifdef KLYGO_USE_CUDA
        auto allocator = std::make_shared<CUDAAllocator>();
        std::size_t bytes = M * N * dtype_size(res_dtype);
        auto storage = std::make_shared<Storage>(bytes, allocator);
        auto impl = std::make_shared<TensorImpl>(storage, std::vector<std::size_t>{M, N}, res_dtype, device());
        Tensor res(impl);
        Tensor a_c = to_contiguous(*this);
        Tensor b_c = to_contiguous(other);
        launch_matmul_cuda(a_c.data(), b_c.data(), res.data(), M, K, N, res_dtype);
        return res;
#else
        throw std::runtime_error("CUDA is not enabled in this build.");
#endif
    }
    
    // Thực thi trên CPU với vòng lặp đa luồng OpenMP r-k-c
    Tensor res = zeros({M, N}, res_dtype);
    Tensor a_c = to_contiguous(*this);
    Tensor b_c = to_contiguous(other);
    
    if (a_c.dtype() == DType::Float32 && b_c.dtype() == DType::Float32 && res_dtype == DType::Float32) {
        matmul_typed(a_c.data<float>(), b_c.data<float>(), res.data<float>(), M, K, N);
    } else if (a_c.dtype() == DType::Float64 && b_c.dtype() == DType::Float64 && res_dtype == DType::Float64) {
        matmul_typed(a_c.data<double>(), b_c.data<double>(), res.data<double>(), M, K, N);
    } else {
        auto get_value = [](const Tensor& t, std::size_t r, std::size_t c) -> double {
            std::size_t idx = r * t.stride()[0] + c * t.stride()[1];
            switch (t.dtype()) {
                case DType::Float32: return static_cast<double>(t.data<float>()[idx]);
                case DType::Float64: return t.data<double>()[idx];
                case DType::Int32:   return static_cast<double>(t.data<int32_t>()[idx]);
                case DType::Int64:   return static_cast<double>(t.data<int64_t>()[idx]);
                case DType::Bool:    return static_cast<double>(t.data<bool>()[idx]);
                default: return 0.0;
            }
        };
        
        auto set_value = [](Tensor& t, std::size_t r, std::size_t c, double val) {
            std::size_t idx = r * t.stride()[0] + c * t.stride()[1];
            switch (t.dtype()) {
                case DType::Float32: t.data<float>()[idx] = static_cast<float>(val); break;
                case DType::Float64: t.data<double>()[idx] = val; break;
                case DType::Int32:   t.data<int32_t>()[idx] = static_cast<int32_t>(val); break;
                case DType::Int64:   t.data<int64_t>()[idx] = static_cast<int64_t>(val); break;
                case DType::Bool:    t.data<bool>()[idx] = static_cast<bool>(val); break;
            }
        };
        
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < static_cast<int64_t>(M); ++r) {
            for (std::size_t k = 0; k < K; ++k) {
                double val_a = get_value(a_c, r, k);
                for (std::size_t c = 0; c < N; ++c) {
                    double current = get_value(res, r, c);
                    set_value(res, r, c, current + val_a * get_value(b_c, k, c));
                }
            }
        }
    }
    
    return res;
}
