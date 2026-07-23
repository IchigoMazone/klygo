#pragma once
#include <cstddef>
#include <cstdint>

enum class DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool
};


inline std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Float64: return sizeof(double);
        case DType::Int32: return sizeof(int32_t);
        case DType::Int64: return sizeof(int64_t);
        case DType::Bool: return sizeof(bool);
        default: return 0;
    }
}