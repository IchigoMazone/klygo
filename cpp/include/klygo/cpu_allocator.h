#pragma once
#include <cstddef>
#include "klygo/allocator.h"

/**
 * @brief Lớp cấp phát bộ nhớ RAM trên CPU với chuẩn căn chỉnh 64-byte.
 * Hỗ trợ tối ưu hóa xử lý chỉ thị SIMD AVX-512 và AVX2 tốc độ cao.
 */
class CPUAllocator : public Allocator {
public:
    /// Cấp phát bộ nhớ RAM căn chỉnh 64-byte
    void* allocate(size_t bytes) override;

    /// Giải phóng bộ nhớ RAM đã căn chỉnh
    void deallocate(void* ptr) override;
};