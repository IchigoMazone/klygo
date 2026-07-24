#pragma once

#include <cstddef>
#include "klygo/dtype.h"

/**
 * @file cuda_kernels.h
 * @brief Khai báo các hàm C++ Launcher gọi tới CUDA Kernels thực thi trên GPU.
 */

#ifdef KLYGO_USE_CUDA
// === Phép toán hai Tensor trên GPU (Binary Tensor Ops) ===
void launch_add_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_sub_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_mul_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_div_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);

// === Phép so sánh hai Tensor trên GPU (Comparison Tensor Ops) ===
void launch_eq_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_ne_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_lt_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_le_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_gt_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_ge_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);

// === Phép toán Tensor với Số thực trên GPU (Scalar Tensor Ops) ===
void launch_add_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype);
void launch_sub_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype);
void launch_mul_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype);
void launch_div_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype);

// === Phép so sánh Tensor với Số thực trên GPU (Comparison Scalar Ops) ===
void launch_eq_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype);
void launch_ne_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype);
void launch_lt_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype);
void launch_le_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype);
void launch_gt_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype);
void launch_ge_scalar_cuda(const void* a, double b, void* c, std::size_t n, DType dtype);

// === Phép toán Đơn biến trên GPU (Unary Tensor Ops) ===
void launch_pow_scalar_cuda(const void* a, double exponent, void* c, std::size_t n, DType dtype);
void launch_sqrt_cuda(const void* a, void* c, std::size_t n, DType dtype);
void launch_exp_cuda(const void* a, void* c, std::size_t n, DType dtype);
void launch_log_cuda(const void* a, void* c, std::size_t n, DType dtype);
void launch_abs_cuda(const void* a, void* c, std::size_t n, DType dtype);
void launch_neg_cuda(const void* a, void* c, std::size_t n, DType dtype);
void launch_clamp_cuda(const void* a, double min_val, double max_val, void* c, std::size_t n, DType dtype);

// === Phép Nhân Ma trận Tiled 2D trên GPU (Matmul GPU) ===
void launch_matmul_cuda(const void* a, const void* b, void* c, std::size_t M, std::size_t K, std::size_t N, DType dtype);
#else
// Mock functions khi build chế độ CPU-only (không có CUDA)
inline void launch_add_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_sub_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_mul_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_div_cuda(const void*, const void*, void*, std::size_t, DType) {}

inline void launch_eq_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_ne_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_lt_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_le_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_gt_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_ge_cuda(const void*, const void*, void*, std::size_t, DType) {}

inline void launch_add_scalar_cuda(const void*, double, void*, std::size_t, DType) {}
inline void launch_sub_scalar_cuda(const void*, double, void*, std::size_t, DType) {}
inline void launch_mul_scalar_cuda(const void*, double, void*, std::size_t, DType) {}
inline void launch_div_scalar_cuda(const void*, double, void*, std::size_t, DType) {}

inline void launch_eq_scalar_cuda(const void*, double, void*, std::size_t, DType) {}
inline void launch_ne_scalar_cuda(const void*, double, void*, std::size_t, DType) {}
inline void launch_lt_scalar_cuda(const void*, double, void*, std::size_t, DType) {}
inline void launch_le_scalar_cuda(const void*, double, void*, std::size_t, DType) {}
inline void launch_gt_scalar_cuda(const void*, double, void*, std::size_t, DType) {}
inline void launch_ge_scalar_cuda(const void*, double, void*, std::size_t, DType) {}

inline void launch_pow_scalar_cuda(const void*, double, void*, std::size_t, DType) {}
inline void launch_sqrt_cuda(const void*, void*, std::size_t, DType) {}
inline void launch_exp_cuda(const void*, void*, std::size_t, DType) {}
inline void launch_log_cuda(const void*, void*, std::size_t, DType) {}
inline void launch_abs_cuda(const void*, void*, std::size_t, DType) {}
inline void launch_neg_cuda(const void*, void*, std::size_t, DType) {}
inline void launch_clamp_cuda(const void*, double, double, void*, std::size_t, DType) {}

inline void launch_matmul_cuda(const void*, const void*, void*, std::size_t, std::size_t, std::size_t, DType) {}
#endif
