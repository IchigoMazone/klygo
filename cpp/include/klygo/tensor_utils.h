#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "klygo/tensor.h"
#include "klygo/device.h"
#include "klygo/cuda_allocator.h"
#include "klygo/cpu_allocator.h"
#include "klygo/cuda_kernels.h"
#include "klygo/tensor_factory.h"

/**
 * @namespace klygo_internal
 * @brief Namespace chứa các hàm tiện ích và các bộ phân giải Template Dispatcher nội bộ của Klygo.
 */
namespace klygo_internal {

/**
 * @brief Kiểm tra xem dữ liệu của Tensor có liên tiếp trong bộ nhớ (contiguous) hay không.
 * @param t Tensor cần kiểm tra
 * @return true Nếu dữ liệu lưu trữ theo đúng thứ tự stride mặc định
 */
inline bool is_contiguous(const Tensor& t) {
    if (t.dim() == 0) return true;
    std::size_t expected_stride = 1;
    for (int d = static_cast<int>(t.dim()) - 1; d >= 0; --d) {
        if (t.shape()[d] == 1) continue;
        if (t.stride()[d] != expected_stride) return false;
        expected_stride *= t.shape()[d];
    }
    return true;
}

/**
 * @brief Sao chép dữ liệu từ Tensor không liên tiếp (sliced/strided) sang mảng bộ nhớ phẳng liên tiếp.
 * Phân luồng đa nhân OpenMP kết hợp ép xung vector SIMD.
 */
template<typename T>
void copy_non_contiguous(const Tensor& src, T* dst_ptr) {
    std::size_t ndim = src.dim();
    if (ndim == 0) {
        dst_ptr[0] = *static_cast<T*>(src.data());
        return;
    }
    std::size_t numel = src.numel();
    const std::vector<std::size_t>& shape = src.shape();
    const std::vector<std::size_t>& stride = src.stride();
    T* src_data = static_cast<T*>(src.data());

#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(numel); ++i) {
        std::size_t idx = static_cast<std::size_t>(i);
        std::size_t flat_idx = 0;
        std::size_t rem = idx;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            std::size_t coord = rem % shape[d];
            rem /= shape[d];
            flat_idx += coord * stride[d];
        }
        dst_ptr[idx] = src_data[flat_idx];
    }
}

/**
 * @brief Chuyển đổi một Tensor bất kỳ thành Tensor có bộ nhớ liên tiếp.
 */
inline Tensor to_contiguous(const Tensor& t) {
    if (is_contiguous(t)) {
        return t;
    }
    Tensor res = empty(t.shape(), t.dtype());
    switch (t.dtype()) {
        case DType::Float32: copy_non_contiguous<float>(t, res.data<float>()); break;
        case DType::Float64: copy_non_contiguous<double>(t, res.data<double>()); break;
        case DType::Int32:   copy_non_contiguous<int32_t>(t, res.data<int32_t>()); break;
        case DType::Int64:   copy_non_contiguous<int64_t>(t, res.data<int64_t>()); break;
        case DType::Bool:    copy_non_contiguous<bool>(t, res.data<bool>()); break;
    }
    return res;
}

/**
 * @brief Quảng bá (promote) kiểu dữ liệu giữa hai DType để chọn kiểu chung phù hợp nhất.
 */
inline DType promote_types(DType t1, DType t2) {
    if (t1 == DType::Float64 || t2 == DType::Float64) return DType::Float64;
    if (t1 == DType::Float32 || t2 == DType::Float32) return DType::Float32;
    if (t1 == DType::Int64 || t2 == DType::Int64) return DType::Int64;
    if (t1 == DType::Int32 || t2 == DType::Int32) return DType::Int32;
    return DType::Bool;
}

/**
 * @brief Vòng lặp tính toán hai con trỏ mảng cùng kiểu dữ liệu trên CPU.
 * Tối ưu hóa phân luồng OpenMP SIMD tự động mở rộng AVX2/AVX-512.
 */
template<typename T, typename Op>
void apply_binary_op_typed(const T* a_ptr, const T* b_ptr, T* res_ptr, std::size_t n, Op op) {
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        res_ptr[i] = op(a_ptr[i], b_ptr[i]);
    }
}

/**
 * @brief Phân giải phép toán hai Tensor trên CPU (Binary Op Dispatcher).
 * Phân loại kiểu dữ liệu ở vòng ngoài để tối ưu hiệu năng.
 */
template<typename Op>
void apply_binary_op(const Tensor& a, const Tensor& b, Tensor& res, Op op) {
    std::size_t n = a.numel();
    if (a.dtype() == b.dtype() && a.dtype() == res.dtype()) {
        switch (a.dtype()) {
            case DType::Float32: apply_binary_op_typed(a.data<float>(), b.data<float>(), res.data<float>(), n, op); return;
            case DType::Float64: apply_binary_op_typed(a.data<double>(), b.data<double>(), res.data<double>(), n, op); return;
            case DType::Int32:   apply_binary_op_typed(a.data<int32_t>(), b.data<int32_t>(), res.data<int32_t>(), n, op); return;
            case DType::Int64:   apply_binary_op_typed(a.data<int64_t>(), b.data<int64_t>(), res.data<int64_t>(), n, op); return;
            case DType::Bool:    apply_binary_op_typed(a.data<bool>(), b.data<bool>(), res.data<bool>(), n, op); return;
        }
    }
    
    // Dự phòng cho trường hợp khác kiểu dữ liệu (Mixed dtype fallback)
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32:   return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64:   return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool:    return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    
    auto set_value = [](Tensor& t, std::size_t i, double val) {
        switch (t.dtype()) {
            case DType::Float32: t.data<float>()[i] = static_cast<float>(val); break;
            case DType::Float64: t.data<double>()[i] = val; break;
            case DType::Int32:   t.data<int32_t>()[i] = static_cast<int32_t>(val); break;
            case DType::Int64:   t.data<int64_t>()[i] = static_cast<int64_t>(val); break;
            case DType::Bool:    t.data<bool>()[i] = static_cast<bool>(val); break;
        }
    };

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        double val_a = get_value(a, i);
        double val_b = get_value(b, i);
        set_value(res, i, op(val_a, val_b));
    }
}

/**
 * @brief Vòng lặp tính toán Tensor với một số thực Scalar trên CPU.
 */
template<typename T, typename Op>
void apply_scalar_op_typed(const T* a_ptr, double val_b, T* res_ptr, std::size_t n, Op op) {
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        res_ptr[i] = static_cast<T>(op(static_cast<double>(a_ptr[i]), val_b));
    }
}

/**
 * @brief Phân giải phép toán Tensor với Số thực (Scalar Op Dispatcher).
 */
template<typename Op>
void apply_scalar_op(const Tensor& a, double val_b, Tensor& res, Op op) {
    std::size_t n = a.numel();
    switch (a.dtype()) {
        case DType::Float32: apply_scalar_op_typed(a.data<float>(), val_b, res.data<float>(), n, op); return;
        case DType::Float64: apply_scalar_op_typed(a.data<double>(), val_b, res.data<double>(), n, op); return;
        case DType::Int32:   apply_scalar_op_typed(a.data<int32_t>(), val_b, res.data<int32_t>(), n, op); return;
        case DType::Int64:   apply_scalar_op_typed(a.data<int64_t>(), val_b, res.data<int64_t>(), n, op); return;
        case DType::Bool:    apply_scalar_op_typed(a.data<bool>(), val_b, res.data<bool>(), n, op); return;
    }
}

/**
 * @brief Vòng lặp tính toán phép toán đơn biến trên CPU (Unary Op Loop).
 */
template<typename T, typename Op>
void apply_unary_op_typed(const T* a_ptr, T* res_ptr, std::size_t n, Op op) {
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        res_ptr[i] = static_cast<T>(op(static_cast<double>(a_ptr[i])));
    }
}

/**
 * @brief Phân giải phép toán đơn biến trên CPU (Unary Op Dispatcher).
 */
template<typename Op>
Tensor unary_op_dispatch(const Tensor& self, Op op) {
    if (self.device().type() == DeviceType::CUDA) {
#ifdef KLYGO_USE_CUDA
        auto allocator = std::make_shared<CUDAAllocator>();
        std::size_t bytes = self.numel() * dtype_size(self.dtype());
        auto storage = std::make_shared<Storage>(bytes, allocator);
        auto impl = std::make_shared<TensorImpl>(storage, self.shape(), self.dtype(), self.device());
        Tensor res(impl);
        return res;
#else
        throw std::runtime_error("CUDA is not enabled in this build.");
#endif
    }

    Tensor a_c = to_contiguous(self);
    Tensor res = empty(self.shape(), self.dtype());
    std::size_t n = self.numel();
    switch (self.dtype()) {
        case DType::Float32: apply_unary_op_typed(a_c.data<float>(), res.data<float>(), n, op); break;
        case DType::Float64: apply_unary_op_typed(a_c.data<double>(), res.data<double>(), n, op); break;
        case DType::Int32:   apply_unary_op_typed(a_c.data<int32_t>(), res.data<int32_t>(), n, op); break;
        case DType::Int64:   apply_unary_op_typed(a_c.data<int64_t>(), res.data<int64_t>(), n, op); break;
        case DType::Bool:    apply_unary_op_typed(a_c.data<bool>(), res.data<bool>(), n, op); break;
    }
    return res;
}

/**
 * @brief Phân giải phép toán binary giữa hai Tensor (tự động chuyển hướng giữa CPU và GPU CUDA).
 */
template<typename CPUOp, typename CUDALauncher>
Tensor binary_op_dispatch(const Tensor& self, const Tensor& other, CPUOp cpu_op, CUDALauncher cuda_launcher) {
    if (self.shape() != other.shape()) throw std::runtime_error("Shape mismatch");
    if (self.device() != other.device()) throw std::runtime_error("Devices must match");
    DType res_dtype = promote_types(self.dtype(), other.dtype());
    
    if (self.device().type() == DeviceType::CUDA) {
#ifdef KLYGO_USE_CUDA
        auto allocator = std::make_shared<CUDAAllocator>();
        std::size_t bytes = self.numel() * dtype_size(res_dtype);
        auto storage = std::make_shared<Storage>(bytes, allocator);
        auto impl = std::make_shared<TensorImpl>(storage, self.shape(), res_dtype, self.device());
        Tensor res(impl);
        cuda_launcher(self.data(), other.data(), res.data(), self.numel(), res_dtype);
        return res;
#else
        throw std::runtime_error("CUDA is not enabled in this build.");
#endif
    } else {
        Tensor a_c = to_contiguous(self);
        Tensor b_c = to_contiguous(other);
        Tensor res = empty(self.shape(), res_dtype);
        apply_binary_op(a_c, b_c, res, cpu_op);
        return res;
    }
}

/**
 * @brief Phân giải phép toán Tensor với Số thực (tự động chuyển hướng giữa CPU và GPU CUDA).
 */
template<typename CPUOp, typename CUDAScalarLauncher>
Tensor scalar_op_dispatch_cuda(const Tensor& self, double other, CPUOp cpu_op, CUDAScalarLauncher cuda_scalar_launcher) {
    if (self.device().type() == DeviceType::CUDA) {
#ifdef KLYGO_USE_CUDA
        auto allocator = std::make_shared<CUDAAllocator>();
        std::size_t bytes = self.numel() * dtype_size(self.dtype());
        auto storage = std::make_shared<Storage>(bytes, allocator);
        auto impl = std::make_shared<TensorImpl>(storage, self.shape(), self.dtype(), self.device());
        Tensor res(impl);
        cuda_scalar_launcher(self.data(), other, res.data(), self.numel(), self.dtype());
        return res;
#else
        throw std::runtime_error("CUDA is not enabled in this build.");
#endif
    } else {
        Tensor a_c = to_contiguous(self);
        Tensor res = empty(self.shape(), self.dtype());
        apply_scalar_op(a_c, other, res, cpu_op);
        return res;
    }
}

template<typename CPUOp>
Tensor scalar_op_dispatch(const Tensor& self, double other, CPUOp cpu_op) {
    return scalar_op_dispatch_cuda(self, other, cpu_op, [](const void*, double, void*, std::size_t, DType){});
}

/**
 * @brief Phân giải phép toán đơn biến (tự động chuyển hướng giữa CPU và GPU CUDA).
 */
template<typename Op, typename CUDALauncher>
Tensor unary_op_dispatch_cuda(const Tensor& self, Op op, CUDALauncher cuda_launcher) {
    if (self.device().type() == DeviceType::CUDA) {
#ifdef KLYGO_USE_CUDA
        auto allocator = std::make_shared<CUDAAllocator>();
        std::size_t bytes = self.numel() * dtype_size(self.dtype());
        auto storage = std::make_shared<Storage>(bytes, allocator);
        auto impl = std::make_shared<TensorImpl>(storage, self.shape(), self.dtype(), self.device());
        Tensor res(impl);
        cuda_launcher(self.data(), res.data(), self.numel(), self.dtype());
        return res;
#else
        throw std::runtime_error("CUDA is not enabled in this build.");
#endif
    }
    return unary_op_dispatch(self, op);
}

} // namespace klygo_internal
