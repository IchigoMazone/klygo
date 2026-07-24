#pragma once
#include <cstddef>
#include <cstdint>

/**
 * @brief Enum định nghĩa các kiểu dữ liệu hỗ trợ trong Tensor Klygo.
 * Tương đương với torch.dtype trong PyTorch.
 */
enum class DType {
    Float32,  // Số thực 32-bit (float)
    Float64,  // Số thực 64-bit (double)
    Int32,    // Số nguyên 32-bit (int32_t)
    Int64,    // Số nguyên 64-bit (int64_t)
    Bool      // Kiểu logic boolean (bool)
};

/**
 * @brief Lấy kích thước theo byte của kiểu dữ liệu DType tương ứng.
 * 
 * @param dtype Kiểu dữ liệu DType
 * @return std::size_t Kích thước tính bằng byte (ví dụ: Float32 -> 4 bytes)
 */
inline std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Float64: return sizeof(double);
        case DType::Int32:   return sizeof(int32_t);
        case DType::Int64:   return sizeof(int64_t);
        case DType::Bool:    return sizeof(bool);
        default: return 0;
    }
}