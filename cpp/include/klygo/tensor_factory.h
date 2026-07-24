#pragma once

#include <vector>
#include "klygo/tensor.h"
#include "klygo/dtype.h"

/**
 * @brief Khởi tạo một Tensor mới với bộ nhớ rác chưa khởi tạo (rỗng).
 * Tương đương với torch.empty() trong PyTorch.
 */
Tensor empty(const std::vector<std::size_t>& shape, DType dtype);

/**
 * @brief Khởi tạo Tensor chứa toàn số 0 (zeros).
 * Tương đương với torch.zeros() trong PyTorch.
 */
Tensor zeros(const std::vector<std::size_t>& shape, DType dtype);

/**
 * @brief Khởi tạo Tensor chứa toàn số 1 (ones).
 * Tương đương với torch.ones() trong PyTorch.
 */
Tensor ones(const std::vector<std::size_t>& shape, DType dtype);

/**
 * @brief Khởi tạo Tensor với tất cả các phần tử bằng giá trị value cho trước.
 * Tương đương với torch.full() trong PyTorch.
 */
Tensor full(const std::vector<std::size_t>& shape, double value, DType dtype);

/**
 * @brief Khởi tạo ma trận đơn vị 2D (Identity Matrix) đường chéo bằng 1.
 * Tương đương với torch.eye() trong PyTorch.
 */
Tensor eye(std::size_t n, std::size_t m, DType dtype);

/**
 * @brief Khởi tạo Tensor 1D chứa dãy số tăng dần từ start tới end theo bước nhảy step.
 * Tương đương với torch.arange() trong PyTorch.
 */
Tensor arange(double start, double end, double step, DType dtype);

/**
 * @brief Khởi tạo Tensor 1D chia đều dải giá trị từ start tới end thành steps phần tử.
 * Tương đương với torch.linspace() trong PyTorch.
 */
Tensor linspace(double start, double end, std::size_t steps, DType dtype);

/**
 * @brief Khởi tạo Tensor chứa các giá trị ngẫu nhiên phân bố đều [0, 1).
 * Tương đương với torch.rand() trong PyTorch.
 */
Tensor rand(const std::vector<std::size_t>& shape, DType dtype);

/**
 * @brief Khởi tạo Tensor chứa các giá trị ngẫu nhiên phân bố chuẩn N(0, 1).
 * Tương đương với torch.randn() trong PyTorch.
 */
Tensor randn(const std::vector<std::size_t>& shape, DType dtype);

/**
 * @brief Khởi tạo Tensor chứa số nguyên ngẫu nhiên trong khoảng [low, high).
 * Tương đương với torch.randint() trong PyTorch.
 */
Tensor randint(int64_t low, int64_t high, const std::vector<std::size_t>& shape, DType dtype);

/// Khởi tạo Tensor empty có cùng kích thước và kiểu dữ liệu như Tensor đầu vào
Tensor empty_like(const Tensor& input, DType dtype);

/// Khởi tạo Tensor zeros có cùng kích thước và kiểu dữ liệu như Tensor đầu vào
Tensor zeros_like(const Tensor& input, DType dtype);

/// Khởi tạo Tensor ones có cùng kích thước và kiểu dữ liệu như Tensor đầu vào
Tensor ones_like(const Tensor& input, DType dtype);

/// Khởi tạo Tensor full có cùng kích thước và kiểu dữ liệu như Tensor đầu vào
Tensor full_like(const Tensor& input, double value, DType dtype);

/// Thiết lập hạt giống ngẫu nhiên cố định (random seed)
void manual_seed(uint64_t seed);

/// Nối danh sách các Tensor theo chiều dim chỉ định (Concat / Cat)
Tensor cat(const std::vector<Tensor>& tensors, int64_t dim = 0);

/// Xếp chồng danh sách các Tensor theo một chiều mới (Stack)
Tensor stack(const std::vector<Tensor>& tensors, int64_t dim = 0);

/// Tách Tensor thành các mảng con theo kích thước split_size
std::vector<Tensor> split(const Tensor& tensor, std::size_t split_size, int64_t dim = 0);

/// Tách Tensor thành số lượng chunks mảng con bằng nhau
std::vector<Tensor> chunk(const Tensor& tensor, std::size_t chunks, int64_t dim = 0);

/// Phép toán chọn phần tử theo điều kiện boolean (Where)
Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other);