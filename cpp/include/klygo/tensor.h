#pragma once

#include <ostream>
#include <cstddef>
#include <memory>
#include <vector>

#include "klygo/tensor_impl.h"
#include "klygo/tensor_printer.h"
#include "klygo/device.h"

/**
 * @brief Lớp Tensor đại diện cho mảng đa chiều chính của Klygo.
 * Giao diện hoàn toàn tương thích với torch.Tensor của PyTorch.
 */
class Tensor {

public:
    /// Khởi tạo Tensor từ một con trỏ TensorImpl
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    /// Lấy tổng số phần tử (numel)
    std::size_t numel() const;
    /// Lấy số chiều (dim / ndim)
    std::size_t dim() const;
    /// Lấy kích thước của một chiều
    std::size_t size(std::size_t dim) const;
    /// Lấy offset bộ nhớ
    std::size_t offset() const;
    /// Lấy mảng kích thước (shape)
    const std::vector<std::size_t>& shape() const;
    /// Lấy mảng bước nhảy (stride)
    const std::vector<std::size_t>& stride() const;
    /// Lấy kiểu dữ liệu (dtype)
    DType dtype() const;

    /// Con trỏ dữ liệu mẫu với ép kiểu tùy chỉnh
    template<typename T>
    T* data() const {
        return static_cast<T*>(impl_->data());
    }

    /// Con trỏ dữ liệu thô (void*)
    void* data() const;
    /// Đối tượng chứa bộ nhớ Storage
    std::shared_ptr<Storage> storage() const;

    /// Đối tượng triển khai nội bộ TensorImpl
    std::shared_ptr<TensorImpl> impl() const;

    /// Thiết bị thực thi (CPU hoặc CUDA)
    Device device() const;
    /// Chuyển đổi Tensor sang thiết bị phần cứng khác (to device)
    Tensor to(const Device& device) const;

    // === Bố cục Bộ nhớ (Memory & Layout) ===
    /// Kiểm tra bộ nhớ Tensor có liên tiếp hay không
    bool is_contiguous() const;
    /// Chuyển đổi Tensor về dạng bộ nhớ liên tiếp
    Tensor contiguous() const;
    /// Bật bản sao độc lập dữ liệu (clone)
    Tensor clone() const;
    /// Kích thước theo byte của 1 phần tử
    std::size_t element_size() const;
    /// Bước nhảy stride của chiều dim
    std::size_t stride(std::size_t dim) const;
    /// Lấy giá trị duy nhất dưới dạng double (khi Tensor có 1 phần tử)
    double item_double() const;

    // === Thao tác Biến đổi Kích thước (Shape Manipulation) ===
    /// Đổi dạng kích thước Tensor (View)
    Tensor view(const std::vector<int64_t>& shape) const;
    /// Hoán đổi hai chiều chỉ định (Transpose)
    Tensor transpose(std::size_t dim0, std::size_t dim1) const;
    /// Transpose ma trận 2D (t())
    Tensor t() const;
    /// Loại bỏ tất cả các chiều có kích thước bằng 1 (Squeeze)
    Tensor squeeze() const;
    /// Loại bỏ chiều dim chỉ định nếu kích thước bằng 1
    Tensor squeeze(int64_t dim) const;
    /// Thêm một chiều mới có kích thước bằng 1 tại vị trí dim (Unsqueeze)
    Tensor unsqueeze(int64_t dim) const;
    /// Hoán đổi thứ tự các chiều theo dims (Permute)
    Tensor permute(const std::vector<int64_t>& dims) const;
    /// Làm phẳng Tensor thành 1D hoặc trong dải chiều (Flatten)
    Tensor flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;

    // === Phép toán Đại số (Out-of-place Arithmetic) ===
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor add(double other) const;
    Tensor sub(double other) const;
    Tensor mul(double other) const;
    Tensor div(double other) const;

    // === Phép toán Trực tiếp trên Bộ nhớ (In-place Arithmetic) ===
    Tensor& add_(const Tensor& other);
    Tensor& add_(double other);
    Tensor& sub_(const Tensor& other);
    Tensor& sub_(double other);
    Tensor& mul_(const Tensor& other);
    Tensor& mul_(double other);
    Tensor& div_(const Tensor& other);
    Tensor& div_(double other);
    Tensor& fill_(double value);
    Tensor& zero_();

    // === Phép toán Đơn biến & Toán học Element-wise ===
    Tensor pow(double exponent) const;
    Tensor pow(const Tensor& exponent) const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor abs() const;
    Tensor neg() const;
    Tensor clamp(double min_val, double max_val) const;

    // === Phép toán So sánh (Comparisons) ===
    Tensor eq(const Tensor& other) const;
    Tensor eq(double other) const;
    Tensor ne(const Tensor& other) const;
    Tensor ne(double other) const;
    Tensor lt(const Tensor& other) const;
    Tensor lt(double other) const;
    Tensor le(const Tensor& other) const;
    Tensor le(double other) const;
    Tensor gt(const Tensor& other) const;
    Tensor gt(double other) const;
    Tensor ge(const Tensor& other) const;
    Tensor ge(double other) const;

    // === Phép thu gọn & Thống kê (Reductions & Stats) ===
    Tensor sum() const;
    Tensor sum(int64_t dim, bool keepdim = false) const;
    Tensor mean() const;
    Tensor mean(int64_t dim, bool keepdim = false) const;
    Tensor max() const;
    std::pair<Tensor, Tensor> max(int64_t dim, bool keepdim = false) const;
    Tensor min() const;
    std::pair<Tensor, Tensor> min(int64_t dim, bool keepdim = false) const;
    Tensor argmax(int64_t dim = -1, bool keepdim = false) const;
    Tensor argmin(int64_t dim = -1, bool keepdim = false) const;
    Tensor prod() const;
    Tensor prod(int64_t dim, bool keepdim = false) const;
    Tensor var(bool unbiased = true) const;
    Tensor var(int64_t dim, bool unbiased = true, bool keepdim = false) const;
    Tensor std(bool unbiased = true) const;
    Tensor std(int64_t dim, bool unbiased = true, bool keepdim = false) const;
    Tensor all() const;
    Tensor all(int64_t dim, bool keepdim = false) const;
    Tensor any() const;
    Tensor any(int64_t dim, bool keepdim = false) const;

    // === Phép Nhân Ma trận (Matrix Multiplication) ===
    Tensor matmul(const Tensor& other) const;

private:
    std::shared_ptr<TensorImpl> impl_; // Con trỏ thực thi nội bộ
};

/// Ghi đè toán tử << để in Tensor
std::ostream& operator<<(
    std::ostream& os,
    const Tensor& tensor
);