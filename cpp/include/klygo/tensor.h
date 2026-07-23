#pragma once

#include <ostream>
#include <cstddef>
#include <memory>

#include "klygo/tensor_impl.h"
#include "klygo/tensor_printer.h"
#include "klygo/device.h"

class Tensor {

public:
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    std::size_t numel() const;
    std::size_t dim() const;
    std::size_t size(std::size_t dim) const;
    std::size_t offset() const;
    const std::vector<std::size_t>& shape() const;
    const std::vector<std::size_t>& stride() const;
    DType dtype() const;
    template<typename T>
    T* data() const {
        return static_cast<T*>(impl_->data());
    }

    void* data() const;
    std::shared_ptr<Storage> storage() const;

    std::shared_ptr<TensorImpl> impl() const;

    Device device() const;
    Tensor to(const Device& device) const;

    // Memory & Layout
    bool is_contiguous() const;
    Tensor contiguous() const;
    Tensor clone() const;
    std::size_t element_size() const;
    std::size_t stride(std::size_t dim) const;
    double item_double() const;

    // Shape Manipulation
    Tensor view(const std::vector<int64_t>& shape) const;
    Tensor transpose(std::size_t dim0, std::size_t dim1) const;
    Tensor t() const;
    Tensor squeeze() const;
    Tensor squeeze(int64_t dim) const;
    Tensor unsqueeze(int64_t dim) const;
    Tensor permute(const std::vector<int64_t>& dims) const;
    Tensor flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;

    // Out-of-place Arithmetic
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor add(double other) const;
    Tensor sub(double other) const;
    Tensor mul(double other) const;
    Tensor div(double other) const;

    // In-place Arithmetic
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

    // Unary & Element-wise Math
    Tensor pow(double exponent) const;
    Tensor pow(const Tensor& exponent) const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor abs() const;
    Tensor neg() const;
    Tensor clamp(double min_val, double max_val) const;

    // Comparisons
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

    // Reductions & Stats
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

    // Matrix Multiplication
    Tensor matmul(const Tensor& other) const;

private:
    std::shared_ptr<TensorImpl> impl_;

};

std::ostream& operator<<(
    std::ostream& os,
    const Tensor& tensor
);