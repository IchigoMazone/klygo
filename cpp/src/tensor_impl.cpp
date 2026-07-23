#include <cstddef>
#include "klygo/tensor_impl.h"

using namespace std;

TensorImpl::TensorImpl(
    shared_ptr<Storage> storage, 
    const vector<size_t>& shape,
    DType dtype,
    Device device
) 
    : 
    storage_(std::move(storage)), 
    shape_(shape),
    dtype_(dtype),
    device_(device)
{
    compute_stride();
}

TensorImpl::TensorImpl(
    shared_ptr<Storage> storage, 
    const vector<size_t>& shape,
    const vector<size_t>& stride,
    size_t offset,
    DType dtype,
    Device device
) 
    : 
    storage_(std::move(storage)), 
    shape_(shape),
    stride_(stride),
    offset_(offset),
    dtype_(dtype),
    device_(device)
{}

void TensorImpl::compute_stride() {
    stride_.resize(shape_.size());

    if (shape_.empty()){
        return;
    }

    stride_.back() = 1;

    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        stride_[i] = stride_[i + 1] * shape_[i + 1];
    }
}

const vector<size_t>& TensorImpl::shape() const {
    return shape_;
}

const vector<size_t>& TensorImpl::stride() const {
    return stride_;
}

void* TensorImpl::data() const {
    if (!storage_ || !storage_->data()) return nullptr;
    char* base = static_cast<char*>(storage_->data());
    return base + offset_ * dtype_size(dtype_);
}

size_t TensorImpl::offset() const {
    return offset_;
}

shared_ptr<Storage> TensorImpl::storage() const {
    return storage_;
}

size_t TensorImpl::numel() const {
    size_t total = 1;
    for (size_t dim : shape_) {
        total *= dim;
    }

    return total; 
}

size_t TensorImpl::dim() const {
    return shape_.size();
}

size_t TensorImpl::size(std::size_t dim) const {
    return shape_[dim];
}

DType TensorImpl::dtype() const {
    return dtype_;
}

Device TensorImpl::device() const {
    return device_;
}

