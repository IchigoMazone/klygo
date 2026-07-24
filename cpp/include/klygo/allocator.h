#pragma once
#include <cstddef>

/**
 * @brief Interface trừu tượng định nghĩa bộ quản lý cấp phát bộ nhớ (Memory Allocator).
 * Cho phép phân tách logic cấp phát bộ nhớ RAM (CPU) và VRAM (CUDA GPU).
 */
class Allocator {
public:
    /**
     * @brief Cấp phát một vùng nhớ với dung lượng tính bằng byte.
     * @param bytes Số byte cần cấp phát
     * @return void* Con trỏ trỏ tới vùng nhớ được cấp phát
     */
    virtual void* allocate(size_t bytes) = 0;

    /**
     * @brief Giải phóng vùng nhớ đã cấp phát.
     * @param ptr Con trỏ trỏ tới vùng nhớ cần giải phóng
     */
    virtual void deallocate(void* ptr) = 0;

    /// Destructor ảo tiêu chuẩn
    virtual ~Allocator() = default;
};