#pragma once
#include <cstddef>
#include <memory>
#include "klygo/allocator.h"

/**
 * @brief Lớp quản lý bộ nhớ byte thô (Raw Byte Storage).
 * Đóng vai trò là bộ chứa dữ liệu thật cho một hoặc nhiều Tensor (chia sẻ bộ nhớ qua shared_ptr).
 */
class Storage {
public:
    /**
     * @brief Khởi tạo vùng nhớ Storage với dung lượng và bộ cấp phát tương ứng.
     * @param bytes Kích thước bộ nhớ tính theo byte
     * @param allocator Bộ cấp phát bộ nhớ (CPUAllocator hoặc CUDAAllocator)
     */
    Storage(std::size_t bytes, std::shared_ptr<Allocator> allocator);

    /// Destructor tự động giải phóng bộ nhớ qua allocator
    ~Storage();

    /// Lấy con trỏ thô trỏ tới vùng dữ liệu
    void* data() const;

    /// Lấy tổng dung lượng bộ nhớ tính bằng byte
    std::size_t bytes() const;

private:
    void* data_ = nullptr;                  // Con trỏ vùng nhớ thô
    std::size_t bytes_ = 0;                 // Dung lượng theo byte
    std::shared_ptr<Allocator> allocator_; // Bộ cấp phát bộ nhớ tương ứng
};