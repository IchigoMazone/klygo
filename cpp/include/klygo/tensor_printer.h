#pragma once

#include <ostream>

class Tensor;

/**
 * @brief Ghi đè toán tử << để in định dạng thông tin và dữ liệu Tensor ra luồng std::ostream.
 * Tương tự hàm print() hoặc repr() trong Python PyTorch.
 */
std::ostream& operator<<(
    std::ostream& os,
    const Tensor& tensor
);