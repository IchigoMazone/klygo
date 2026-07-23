#pragma once

#include <cstddef>
#include "klygo/dtype.h"

#ifdef KLYGO_USE_CUDA
void launch_add_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_sub_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_mul_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
void launch_div_cuda(const void* a, const void* b, void* c, std::size_t n, DType dtype);
#else
inline void launch_add_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_sub_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_mul_cuda(const void*, const void*, void*, std::size_t, DType) {}
inline void launch_div_cuda(const void*, const void*, void*, std::size_t, DType) {}
#endif
