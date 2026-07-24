#include <cstring>
#include <stdexcept>
#include "klygo/tensor.h"
#include "klygo/tensor_utils.h"

using namespace std;
using namespace klygo_internal;

// Khởi tạo Tensor từ đối tượng triển khai TensorImpl
Tensor::Tensor(shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

// Lấy con trỏ TensorImpl
shared_ptr<TensorImpl> Tensor::impl() const {
    return impl_;
}

// Tổng số phần tử trong Tensor
size_t Tensor::numel() const {
    return impl_->numel();
}

// Con trỏ dữ liệu thô
void* Tensor::data() const {
    return impl_->data();
}

// Số chiều của Tensor (ndim)
size_t Tensor::dim() const {
    return impl_->dim();
}

// Kích thước chiều dim
size_t Tensor::size(size_t dim) const {
    return impl_->size(dim);
}

// Mảng kích thước (shape)
const vector<size_t>& Tensor::shape() const {
    return impl_->shape();
}

// Mảng bước nhảy (stride)
const vector<size_t>& Tensor::stride() const {
    return impl_->stride();
}

// Kiểu dữ liệu DType
DType Tensor::dtype() const {
    return impl_->dtype();
}

// Bộ chứa bộ nhớ Storage
shared_ptr<Storage> Tensor::storage() const {
    return impl_->storage();
}

// Offset phân đoạn dữ liệu
std::size_t Tensor::offset() const {
    return impl_->offset();
}

// Thiết bị thực thi (CPU hoặc CUDA)
Device Tensor::device() const {
    return impl_->device();
}

// Sao chép và chuyển đổi Tensor sang thiết bị đích (CPU <-> CUDA GPU)
Tensor Tensor::to(const Device& target_device) const {
    if (device() == target_device) {
        return *this;
    }
    
    std::size_t bytes = numel() * dtype_size(dtype());
    void* src_ptr = data();
    
#ifdef KLYGO_USE_CUDA
    std::shared_ptr<Allocator> allocator;
    if (target_device.type() == DeviceType::CUDA) {
        // Chuyển dữ liệu từ CPU RAM -> GPU VRAM (HostToDevice)
        allocator = std::make_shared<CUDAAllocator>();
        auto storage = std::make_shared<Storage>(bytes, allocator);
        void* dst_ptr = storage->data();
        cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy to GPU failed: ") + cudaGetErrorString(err));
        }
        auto impl = std::make_shared<TensorImpl>(storage, shape(), stride(), offset(), dtype(), target_device);
        return Tensor(impl);
    } else {
        // Chuyển dữ liệu từ GPU VRAM -> CPU RAM (DeviceToHost)
        allocator = std::make_shared<CPUAllocator>();
        auto storage = std::make_shared<Storage>(bytes, allocator);
        void* dst_ptr = storage->data();
        cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy to CPU failed: ") + cudaGetErrorString(err));
        }
        auto impl = std::make_shared<TensorImpl>(storage, shape(), stride(), offset(), dtype(), target_device);
        return Tensor(impl);
    }
#else
    if (target_device.type() == DeviceType::CUDA) {
        throw std::runtime_error("CUDA is not enabled in this build.");
    }
    return *this;
#endif
}

// Kiểm tra tính liên tiếp của bộ nhớ
bool Tensor::is_contiguous() const {
    return klygo_internal::is_contiguous(*this);
}

// Chuyển đổi bộ nhớ về dạng liên tiếp (contiguous)
Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    return to_contiguous(*this);
}

// Tạo bản sao hoàn toàn mới dữ liệu của Tensor (Clone)
Tensor Tensor::clone() const {
    Tensor res = empty(shape(), dtype());
    std::size_t bytes = numel() * dtype_size(dtype());
    std::memcpy(res.data(), data(), bytes);
    if (device().type() != DeviceType::CPU) {
        res = res.to(device());
    }
    return res;
}

// Kích thước byte của 1 phần tử
std::size_t Tensor::element_size() const {
    return dtype_size(dtype());
}

// Stride của chiều dim
std::size_t Tensor::stride(std::size_t dim) const {
    return stride()[dim];
}

// Lấy giá trị duy nhất dưới dạng double (khi Tensor có đúng 1 phần tử)
double Tensor::item_double() const {
    if (numel() != 1) {
        throw std::runtime_error("item() can only be called for 1-element tensors");
    }
    Tensor c = to_contiguous(*this);
    switch (c.dtype()) {
        case DType::Float32: return static_cast<double>(c.data<float>()[0]);
        case DType::Float64: return c.data<double>()[0];
        case DType::Int32:   return static_cast<double>(c.data<int32_t>()[0]);
        case DType::Int64:   return static_cast<double>(c.data<int64_t>()[0]);
        case DType::Bool:    return static_cast<double>(c.data<bool>()[0]);
    }
    return 0.0;
}
