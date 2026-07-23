#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include "klygo/tensor.h"
#include "klygo/device.h"
#include "klygo/cuda_allocator.h"
#include "klygo/cpu_allocator.h"
#include "klygo/cuda_kernels.h"
#include "klygo/tensor_factory.h"

using namespace std;

Tensor::Tensor(shared_ptr<TensorImpl> impl): impl_(std::move(impl)) {}

shared_ptr<TensorImpl> Tensor::impl() const {
    return impl_;
}

size_t Tensor::numel() const {
    return impl_->numel();
}

void* Tensor::data() const {
    return impl_->data();
}

size_t Tensor::dim() const {
    return impl_->dim();
}

size_t Tensor::size(size_t dim) const {
    return impl_->size(dim);
}

const vector<size_t>& Tensor::shape() const {
    return impl_->shape();
}

const vector<size_t>& Tensor::stride() const {
    return impl_->stride();
}

DType Tensor::dtype() const {
    return impl_->dtype();
}

shared_ptr<Storage> Tensor::storage() const {
    return impl_->storage();
}

std::size_t Tensor::offset() const {
    return impl_->offset();
}

Device Tensor::device() const {
    return impl_->device();
}

Tensor Tensor::to(const Device& target_device) const {
    if (device() == target_device) {
        return *this;
    }
    
    std::size_t bytes = numel() * dtype_size(dtype());
    void* src_ptr = data();
    
#ifdef KLYGO_USE_CUDA
    std::shared_ptr<Allocator> allocator;
    if (target_device.type() == DeviceType::CUDA) {
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

Tensor Tensor::view(const std::vector<int64_t>& shape) const {
    std::size_t numel_val = numel();
    std::vector<std::size_t> target_shape;
    int64_t infer_idx = -1;
    std::size_t prod = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            if (infer_idx != -1) {
                throw std::runtime_error("Only one dimension can be -1");
            }
            infer_idx = i;
            target_shape.push_back(0); // placeholder
        } else if (shape[i] < 0) {
            throw std::runtime_error("Dimension cannot be negative (except -1)");
        } else {
            target_shape.push_back(static_cast<std::size_t>(shape[i]));
            prod *= static_cast<std::size_t>(shape[i]);
        }
    }
    
    if (infer_idx != -1) {
        if (prod == 0 || numel_val % prod != 0) {
            throw std::runtime_error("Invalid shape for view");
        }
        target_shape[infer_idx] = numel_val / prod;
    } else {
        std::size_t target_prod = 1;
        for (auto s : target_shape) target_prod *= s;
        if (target_prod != numel_val) {
            throw std::runtime_error("Shape size must match tensor size");
        }
    }

    std::vector<std::size_t> target_stride(target_shape.size());
    if (!target_shape.empty()) {
        target_stride.back() = 1;
        for (int i = static_cast<int>(target_shape.size()) - 2; i >= 0; --i) {
            target_stride[i] = target_stride[i + 1] * target_shape[i + 1];
        }
    }

    auto impl = std::make_shared<TensorImpl>(
        impl_->storage(),
        target_shape,
        target_stride,
        impl_->offset(),
        impl_->dtype()
    );
    return Tensor(impl);
}

Tensor Tensor::transpose(std::size_t dim0, std::size_t dim1) const {
    std::size_t ndim = dim();
    if (dim0 >= ndim || dim1 >= ndim) {
        throw std::out_of_range("Dimension out of range for transpose");
    }
    std::vector<std::size_t> new_shape = shape();
    std::vector<std::size_t> new_stride = stride();
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_stride[dim0], new_stride[dim1]);

    auto impl = std::make_shared<TensorImpl>(
        impl_->storage(),
        new_shape,
        new_stride,
        impl_->offset(),
        impl_->dtype()
    );
    return Tensor(impl);
}

Tensor Tensor::t() const {
    if (dim() > 2) {
        throw std::runtime_error("t() expects a tensor with <= 2 dimensions");
    }
    if (dim() < 2) {
        return *this;
    }
    return transpose(0, 1);
}

namespace {

bool is_contiguous(const Tensor& t) {
    if (t.dim() == 0) return true;
    std::size_t expected_stride = 1;
    for (int d = static_cast<int>(t.dim()) - 1; d >= 0; --d) {
        if (t.shape()[d] == 1) continue;
        if (t.stride()[d] != expected_stride) return false;
        expected_stride *= t.shape()[d];
    }
    return true;
}

template<typename T>
void copy_non_contiguous(const Tensor& src, T* dst_ptr) {
    std::size_t ndim = src.dim();
    if (ndim == 0) {
        dst_ptr[0] = *static_cast<T*>(src.data());
        return;
    }
    std::vector<std::size_t> coords(ndim, 0);
    std::size_t numel = src.numel();
    const std::vector<std::size_t>& shape = src.shape();
    const std::vector<std::size_t>& stride = src.stride();
    T* src_data = static_cast<T*>(src.data());
    
    for (std::size_t i = 0; i < numel; ++i) {
        std::size_t flat_idx = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            flat_idx += coords[d] * stride[d];
        }
        dst_ptr[i] = src_data[flat_idx];
        
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < shape[d]) {
                break;
            }
            coords[d] = 0;
        }
    }
}

} // namespace

namespace {

Tensor to_contiguous(const Tensor& t) {
    if (is_contiguous(t)) {
        return t;
    }
    Tensor res = empty(t.shape(), t.dtype());
    switch (t.dtype()) {
        case DType::Float32: copy_non_contiguous<float>(t, res.data<float>()); break;
        case DType::Float64: copy_non_contiguous<double>(t, res.data<double>()); break;
        case DType::Int32: copy_non_contiguous<int32_t>(t, res.data<int32_t>()); break;
        case DType::Int64: copy_non_contiguous<int64_t>(t, res.data<int64_t>()); break;
        case DType::Bool: copy_non_contiguous<bool>(t, res.data<bool>()); break;
    }
    return res;
}

DType promote_types(DType t1, DType t2) {
    if (t1 == DType::Float64 || t2 == DType::Float64) return DType::Float64;
    if (t1 == DType::Float32 || t2 == DType::Float32) return DType::Float32;
    if (t1 == DType::Int64 || t2 == DType::Int64) return DType::Int64;
    if (t1 == DType::Int32 || t2 == DType::Int32) return DType::Int32;
    return DType::Bool;
}

template<typename Op>
void apply_binary_op(const Tensor& a, const Tensor& b, Tensor& res, Op op) {
    std::size_t n = a.numel();
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    
    auto set_value = [](Tensor& t, std::size_t i, double val) {
        switch (t.dtype()) {
            case DType::Float32: t.data<float>()[i] = static_cast<float>(val); break;
            case DType::Float64: t.data<double>()[i] = val; break;
            case DType::Int32: t.data<int32_t>()[i] = static_cast<int32_t>(val); break;
            case DType::Int64: t.data<int64_t>()[i] = static_cast<int64_t>(val); break;
            case DType::Bool: t.data<bool>()[i] = static_cast<bool>(val); break;
        }
    };

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        double val_a = get_value(a, i);
        double val_b = get_value(b, i);
        set_value(res, i, op(val_a, val_b));
    }
}

template<typename Op>
void apply_scalar_op(const Tensor& a, double val_b, Tensor& res, Op op) {
    std::size_t n = a.numel();
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    
    auto set_value = [](Tensor& t, std::size_t i, double val) {
        switch (t.dtype()) {
            case DType::Float32: t.data<float>()[i] = static_cast<float>(val); break;
            case DType::Float64: t.data<double>()[i] = val; break;
            case DType::Int32: t.data<int32_t>()[i] = static_cast<int32_t>(val); break;
            case DType::Int64: t.data<int64_t>()[i] = static_cast<int64_t>(val); break;
            case DType::Bool: t.data<bool>()[i] = static_cast<bool>(val); break;
        }
    };

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        double val_a = get_value(a, i);
        set_value(res, i, op(val_a, val_b));
    }
}

template<typename CPUOp, typename CUDALauncher>
Tensor binary_op_dispatch(const Tensor& self, const Tensor& other, CPUOp cpu_op, CUDALauncher cuda_launcher) {
    if (self.shape() != other.shape()) throw std::runtime_error("Shape mismatch");
    if (self.device() != other.device()) throw std::runtime_error("Devices must match");
    DType res_dtype = promote_types(self.dtype(), other.dtype());
    
    if (self.device().type() == DeviceType::CUDA) {
#ifdef KLYGO_USE_CUDA
        auto allocator = std::make_shared<CUDAAllocator>();
        std::size_t bytes = self.numel() * dtype_size(res_dtype);
        auto storage = std::make_shared<Storage>(bytes, allocator);
        auto impl = std::make_shared<TensorImpl>(storage, self.shape(), res_dtype, self.device());
        Tensor res(impl);
        cuda_launcher(self.data(), other.data(), res.data(), self.numel(), res_dtype);
        return res;
#else
        throw std::runtime_error("CUDA is not enabled in this build.");
#endif
    } else {
        Tensor a_c = to_contiguous(self);
        Tensor b_c = to_contiguous(other);
        Tensor res = empty(self.shape(), res_dtype);
        apply_binary_op(a_c, b_c, res, cpu_op);
        return res;
    }
}

template<typename CPUOp>
Tensor scalar_op_dispatch(const Tensor& self, double other, CPUOp cpu_op) {
    if (self.device().type() == DeviceType::CUDA) {
        Tensor cpu_tensor = self.to(Device(DeviceType::CPU));
        Tensor res_cpu = empty(self.shape(), self.dtype());
        apply_scalar_op(cpu_tensor, other, res_cpu, cpu_op);
        return res_cpu.to(self.device());
    } else {
        Tensor a_c = to_contiguous(self);
        Tensor res = empty(self.shape(), self.dtype());
        apply_scalar_op(a_c, other, res, cpu_op);
        return res;
    }
}

} // namespace

Tensor Tensor::add(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::plus<double>(), launch_add_cuda);
}

Tensor Tensor::sub(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::minus<double>(), launch_sub_cuda);
}

Tensor Tensor::mul(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::multiplies<double>(), launch_mul_cuda);
}

Tensor Tensor::div(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::divides<double>(), launch_div_cuda);
}

Tensor Tensor::add(double other) const {
    return scalar_op_dispatch(*this, other, std::plus<double>());
}

Tensor Tensor::sub(double other) const {
    return scalar_op_dispatch(*this, other, std::minus<double>());
}

Tensor Tensor::mul(double other) const {
    return scalar_op_dispatch(*this, other, std::multiplies<double>());
}

Tensor Tensor::div(double other) const {
    return scalar_op_dispatch(*this, other, std::divides<double>());
}

Tensor Tensor::sum() const {
    double total = 0.0;
    Tensor self_c = to_contiguous(*this);
    std::size_t n = self_c.numel();
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
#ifdef _OPENMP
#pragma omp parallel for reduction(+:total)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        total += get_value(self_c, i);
    }
    DType res_dtype = (dtype() == DType::Bool) ? DType::Int64 : dtype();
    Tensor res = empty({}, res_dtype);
    switch (res_dtype) {
        case DType::Float32: res.data<float>()[0] = static_cast<float>(total); break;
        case DType::Float64: res.data<double>()[0] = total; break;
        case DType::Int32: res.data<int32_t>()[0] = static_cast<int32_t>(total); break;
        case DType::Int64: res.data<int64_t>()[0] = static_cast<int64_t>(total); break;
        case DType::Bool: res.data<bool>()[0] = static_cast<bool>(total); break;
    }
    return res;
}

Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    std::size_t ndim = this->dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= static_cast<int64_t>(ndim)) {
        throw std::out_of_range("dim out of range");
    }
    
    std::vector<std::size_t> out_shape;
    for (std::size_t i = 0; i < ndim; ++i) {
        if (i == static_cast<std::size_t>(dim)) {
            if (keepdim) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(shape()[i]);
        }
    }
    
    DType res_dtype = (dtype() == DType::Bool) ? DType::Int64 : dtype();
    Tensor res = zeros(out_shape, res_dtype);
    
    Tensor self_c = to_contiguous(*this);
    std::vector<std::size_t> coords(ndim, 0);
    std::size_t numel_val = self_c.numel();
    
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    
    auto add_value = [](Tensor& t, std::size_t i, double val) {
        switch (t.dtype()) {
            case DType::Float32: t.data<float>()[i] += static_cast<float>(val); break;
            case DType::Float64: t.data<double>()[i] += val; break;
            case DType::Int32: t.data<int32_t>()[i] += static_cast<int32_t>(val); break;
            case DType::Int64: t.data<int64_t>()[i] += static_cast<int64_t>(val); break;
            case DType::Bool: t.data<bool>()[i] = static_cast<bool>(static_cast<double>(t.data<bool>()[i]) + val); break;
        }
    };
    
    const std::vector<std::size_t>& res_stride = res.stride();
    
    for (std::size_t i = 0; i < numel_val; ++i) {
        std::size_t out_flat_idx = 0;
        std::size_t out_d = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            if (d == static_cast<std::size_t>(dim)) {
                if (keepdim) {
                    out_d++;
                }
            } else {
                out_flat_idx += coords[d] * res_stride[out_d++];
            }
        }
        
        double val = get_value(self_c, i);
        add_value(res, out_flat_idx, val);
        
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < self_c.shape()[d]) {
                break;
            }
            coords[d] = 0;
        }
    }
    
    return res;
}

Tensor Tensor::mean() const {
    Tensor s = sum();
    return s.div(static_cast<double>(numel()));
}

Tensor Tensor::mean(int64_t dim, bool keepdim) const {
    Tensor s = sum(dim, keepdim);
    std::size_t ndim = this->dim();
    if (dim < 0) dim += ndim;
    return s.div(static_cast<double>(size(dim)));
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (dim() != 2 || other.dim() != 2) {
        throw std::runtime_error("matmul expects 2D tensors");
    }
    if (size(1) != other.size(0)) {
        throw std::runtime_error("Matrix inner dimensions must match");
    }
    
    std::size_t M = size(0);
    std::size_t K = size(1);
    std::size_t N = other.size(1);
    
    DType res_dtype = promote_types(dtype(), other.dtype());
    Tensor res = zeros({M, N}, res_dtype);
    
    Tensor a_c = to_contiguous(*this);
    Tensor b_c = to_contiguous(other);
    
    auto get_value = [](const Tensor& t, std::size_t r, std::size_t c) -> double {
        std::size_t idx = r * t.stride()[0] + c * t.stride()[1];
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[idx]);
            case DType::Float64: return t.data<double>()[idx];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[idx]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[idx]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[idx]);
            default: return 0.0;
        }
    };
    
    auto set_value = [](Tensor& t, std::size_t r, std::size_t c, double val) {
        std::size_t idx = r * t.stride()[0] + c * t.stride()[1];
        switch (t.dtype()) {
            case DType::Float32: t.data<float>()[idx] = static_cast<float>(val); break;
            case DType::Float64: t.data<double>()[idx] = val; break;
            case DType::Int32: t.data<int32_t>()[idx] = static_cast<int32_t>(val); break;
            case DType::Int64: t.data<int64_t>()[idx] = static_cast<int64_t>(val); break;
            case DType::Bool: t.data<bool>()[idx] = static_cast<bool>(val); break;
        }
    };
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t r = 0; r < static_cast<int64_t>(M); ++r) {
        for (int64_t c = 0; c < static_cast<int64_t>(N); ++c) {
            double sum_val = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                sum_val += get_value(a_c, r, k) * get_value(b_c, k, c);
            }
            set_value(res, r, c, sum_val);
        }
    }
    
    return res;
}

// Memory & Layout
bool Tensor::is_contiguous() const {
    return ::is_contiguous(*this);
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    return to_contiguous(*this);
}

Tensor Tensor::clone() const {
    Tensor res = empty(shape(), dtype());
    std::size_t bytes = numel() * dtype_size(dtype());
    std::memcpy(res.data(), data(), bytes);
    if (device().type() != DeviceType::CPU) {
        res = res.to(device());
    }
    return res;
}

std::size_t Tensor::element_size() const {
    return dtype_size(dtype());
}

std::size_t Tensor::stride(std::size_t dim) const {
    return stride()[dim];
}

double Tensor::item_double() const {
    if (numel() != 1) {
        throw std::runtime_error("item() can only be called for 1-element tensors");
    }
    Tensor c = to_contiguous(*this);
    switch (c.dtype()) {
        case DType::Float32: return static_cast<double>(c.data<float>()[0]);
        case DType::Float64: return c.data<double>()[0];
        case DType::Int32: return static_cast<double>(c.data<int32_t>()[0]);
        case DType::Int64: return static_cast<double>(c.data<int64_t>()[0]);
        case DType::Bool: return static_cast<double>(c.data<bool>()[0]);
    }
    return 0.0;
}

// Squeeze & Unsqueeze & Permute & Flatten
Tensor Tensor::squeeze() const {
    std::vector<std::size_t> new_shape;
    std::vector<std::size_t> new_stride;
    for (std::size_t i = 0; i < dim(); ++i) {
        if (shape()[i] != 1) {
            new_shape.push_back(shape()[i]);
            new_stride.push_back(stride()[i]);
        }
    }
    auto impl = std::make_shared<TensorImpl>(storage(), new_shape, new_stride, offset(), dtype(), device());
    return Tensor(impl);
}

Tensor Tensor::squeeze(int64_t dim_idx) const {
    int64_t nd = static_cast<int64_t>(dim());
    if (dim_idx < 0) dim_idx += nd;
    if (dim_idx < 0 || dim_idx >= nd) throw std::out_of_range("dim out of range");
    if (shape()[dim_idx] != 1) return *this;
    
    std::vector<std::size_t> new_shape;
    std::vector<std::size_t> new_stride;
    for (int64_t i = 0; i < nd; ++i) {
        if (i != dim_idx) {
            new_shape.push_back(shape()[i]);
            new_stride.push_back(stride()[i]);
        }
    }
    auto impl = std::make_shared<TensorImpl>(storage(), new_shape, new_stride, offset(), dtype(), device());
    return Tensor(impl);
}

Tensor Tensor::unsqueeze(int64_t dim_idx) const {
    int64_t nd = static_cast<int64_t>(dim());
    if (dim_idx < 0) dim_idx += nd + 1;
    if (dim_idx < 0 || dim_idx > nd + 1) throw std::out_of_range("dim out of range");

    std::vector<std::size_t> new_shape;
    std::vector<std::size_t> new_stride;
    std::size_t src_i = 0;
    for (int64_t i = 0; i <= nd; ++i) {
        if (i == dim_idx) {
            new_shape.push_back(1);
            std::size_t st = (src_i < static_cast<std::size_t>(nd)) ? stride()[src_i] : 1;
            new_stride.push_back(st);
        } else {
            new_shape.push_back(shape()[src_i]);
            new_stride.push_back(stride()[src_i]);
            src_i++;
        }
    }
    auto impl = std::make_shared<TensorImpl>(storage(), new_shape, new_stride, offset(), dtype(), device());
    return Tensor(impl);
}

Tensor Tensor::permute(const std::vector<int64_t>& dims) const {
    if (dims.size() != dim()) throw std::runtime_error("number of dims doesn't match tensor dimension");
    std::vector<std::size_t> new_shape(dim());
    std::vector<std::size_t> new_stride(dim());
    int64_t nd = static_cast<int64_t>(dim());
    
    for (std::size_t i = 0; i < dim(); ++i) {
        int64_t d = dims[i];
        if (d < 0) d += nd;
        if (d < 0 || d >= nd) throw std::out_of_range("dim out of range");
        new_shape[i] = shape()[d];
        new_stride[i] = stride()[d];
    }
    auto impl = std::make_shared<TensorImpl>(storage(), new_shape, new_stride, offset(), dtype(), device());
    return Tensor(impl);
}

Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    int64_t nd = static_cast<int64_t>(dim());
    if (nd == 0) return *this;
    if (start_dim < 0) start_dim += nd;
    if (end_dim < 0) end_dim += nd;
    if (start_dim < 0 || start_dim >= nd || end_dim < 0 || end_dim >= nd || start_dim > end_dim) {
        throw std::out_of_range("invalid start_dim or end_dim");
    }

    std::vector<int64_t> target_shape;
    for (int64_t i = 0; i < start_dim; ++i) target_shape.push_back(static_cast<int64_t>(shape()[i]));
    
    int64_t flat_size = 1;
    for (int64_t i = start_dim; i <= end_dim; ++i) flat_size *= static_cast<int64_t>(shape()[i]);
    target_shape.push_back(flat_size);

    for (int64_t i = end_dim + 1; i < nd; ++i) target_shape.push_back(static_cast<int64_t>(shape()[i]));

    return view(target_shape);
}

// In-place operations
Tensor& Tensor::add_(const Tensor& other) {
    Tensor res = add(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}
Tensor& Tensor::add_(double other) {
    Tensor res = add(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}
Tensor& Tensor::sub_(const Tensor& other) {
    Tensor res = sub(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}
Tensor& Tensor::sub_(double other) {
    Tensor res = sub(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}
Tensor& Tensor::mul_(const Tensor& other) {
    Tensor res = mul(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}
Tensor& Tensor::mul_(double other) {
    Tensor res = mul(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}
Tensor& Tensor::div_(const Tensor& other) {
    Tensor res = div(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}
Tensor& Tensor::div_(double other) {
    Tensor res = div(other);
    std::memcpy(data(), res.data(), numel() * dtype_size(dtype()));
    return *this;
}
Tensor& Tensor::fill_(double value) {
    Tensor f = full(shape(), value, dtype());
    std::memcpy(data(), f.data(), numel() * dtype_size(dtype()));
    return *this;
}
Tensor& Tensor::zero_() {
    return fill_(0.0);
}

// Unary & Element-wise Math
namespace {
template<typename Op>
Tensor unary_op_dispatch(const Tensor& self, Op op) {
    Tensor a_c = to_contiguous(self);
    Tensor res = empty(self.shape(), self.dtype());
    std::size_t n = self.numel();
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    
    auto set_value = [](Tensor& t, std::size_t i, double val) {
        switch (t.dtype()) {
            case DType::Float32: t.data<float>()[i] = static_cast<float>(val); break;
            case DType::Float64: t.data<double>()[i] = val; break;
            case DType::Int32: t.data<int32_t>()[i] = static_cast<int32_t>(val); break;
            case DType::Int64: t.data<int64_t>()[i] = static_cast<int64_t>(val); break;
            case DType::Bool: t.data<bool>()[i] = static_cast<bool>(val); break;
        }
    };

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        set_value(res, i, op(get_value(a_c, i)));
    }
    return res;
}
} // namespace

Tensor Tensor::pow(double exponent) const {
    return unary_op_dispatch(*this, [exponent](double x) { return std::pow(x, exponent); });
}

Tensor Tensor::pow(const Tensor& exponent) const {
    return binary_op_dispatch(*this, exponent, [](double x, double y) { return std::pow(x, y); }, [](const void*, const void*, void*, std::size_t, DType){});
}

Tensor Tensor::sqrt() const {
    return unary_op_dispatch(*this, [](double x) { return std::sqrt(x); });
}

Tensor Tensor::exp() const {
    return unary_op_dispatch(*this, [](double x) { return std::exp(x); });
}

Tensor Tensor::log() const {
    return unary_op_dispatch(*this, [](double x) { return std::log(x); });
}

Tensor Tensor::abs() const {
    return unary_op_dispatch(*this, [](double x) { return std::abs(x); });
}

Tensor Tensor::neg() const {
    return unary_op_dispatch(*this, [](double x) { return -x; });
}

Tensor Tensor::clamp(double min_val, double max_val) const {
    return unary_op_dispatch(*this, [min_val, max_val](double x) {
        return std::max(min_val, std::min(max_val, x));
    });
}

// Comparisons
Tensor Tensor::eq(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::equal_to<double>(), [](const void*, const void*, void*, std::size_t, DType){});
}
Tensor Tensor::eq(double other) const {
    return scalar_op_dispatch(*this, other, std::equal_to<double>());
}
Tensor Tensor::ne(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::not_equal_to<double>(), [](const void*, const void*, void*, std::size_t, DType){});
}
Tensor Tensor::ne(double other) const {
    return scalar_op_dispatch(*this, other, std::not_equal_to<double>());
}
Tensor Tensor::lt(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::less<double>(), [](const void*, const void*, void*, std::size_t, DType){});
}
Tensor Tensor::lt(double other) const {
    return scalar_op_dispatch(*this, other, std::less<double>());
}
Tensor Tensor::le(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::less_equal<double>(), [](const void*, const void*, void*, std::size_t, DType){});
}
Tensor Tensor::le(double other) const {
    return scalar_op_dispatch(*this, other, std::less_equal<double>());
}
Tensor Tensor::gt(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::greater<double>(), [](const void*, const void*, void*, std::size_t, DType){});
}
Tensor Tensor::gt(double other) const {
    return scalar_op_dispatch(*this, other, std::greater<double>());
}
Tensor Tensor::ge(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::greater_equal<double>(), [](const void*, const void*, void*, std::size_t, DType){});
}
Tensor Tensor::ge(double other) const {
    return scalar_op_dispatch(*this, other, std::greater_equal<double>());
}

// Reductions & Stats
Tensor Tensor::prod() const {
    Tensor self_c = to_contiguous(*this);
    double p = 1.0;
    std::size_t n = self_c.numel();
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 1.0;
        }
    };
    for (std::size_t i = 0; i < n; ++i) p *= get_value(self_c, i);
    Tensor res = empty({}, dtype());
    switch (dtype()) {
        case DType::Float32: res.data<float>()[0] = static_cast<float>(p); break;
        case DType::Float64: res.data<double>()[0] = p; break;
        case DType::Int32: res.data<int32_t>()[0] = static_cast<int32_t>(p); break;
        case DType::Int64: res.data<int64_t>()[0] = static_cast<int64_t>(p); break;
        case DType::Bool: res.data<bool>()[0] = static_cast<bool>(p); break;
    }
    return res;
}

Tensor Tensor::max() const {
    Tensor self_c = to_contiguous(*this);
    if (numel() == 0) throw std::runtime_error("cannot perform max on empty tensor");
    double mx = 0.0;
    std::size_t n = self_c.numel();
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    mx = get_value(self_c, 0);
    for (std::size_t i = 1; i < n; ++i) mx = std::max(mx, get_value(self_c, i));
    Tensor res = empty({}, dtype());
    switch (dtype()) {
        case DType::Float32: res.data<float>()[0] = static_cast<float>(mx); break;
        case DType::Float64: res.data<double>()[0] = mx; break;
        case DType::Int32: res.data<int32_t>()[0] = static_cast<int32_t>(mx); break;
        case DType::Int64: res.data<int64_t>()[0] = static_cast<int64_t>(mx); break;
        case DType::Bool: res.data<bool>()[0] = static_cast<bool>(mx); break;
    }
    return res;
}

Tensor Tensor::min() const {
    Tensor self_c = to_contiguous(*this);
    if (numel() == 0) throw std::runtime_error("cannot perform min on empty tensor");
    double mn = 0.0;
    std::size_t n = self_c.numel();
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    mn = get_value(self_c, 0);
    for (std::size_t i = 1; i < n; ++i) mn = std::min(mn, get_value(self_c, i));
    Tensor res = empty({}, dtype());
    switch (dtype()) {
        case DType::Float32: res.data<float>()[0] = static_cast<float>(mn); break;
        case DType::Float64: res.data<double>()[0] = mn; break;
        case DType::Int32: res.data<int32_t>()[0] = static_cast<int32_t>(mn); break;
        case DType::Int64: res.data<int64_t>()[0] = static_cast<int64_t>(mn); break;
        case DType::Bool: res.data<bool>()[0] = static_cast<bool>(mn); break;
    }
    return res;
}

Tensor Tensor::var(bool unbiased) const {
    Tensor m = mean();
    double mean_val = m.item_double();
    Tensor diff = sub(mean_val);
    Tensor sq = diff.mul(diff);
    Tensor s = sq.sum();
    double denom = unbiased ? static_cast<double>(numel() - 1) : static_cast<double>(numel());
    if (denom <= 0) denom = 1.0;
    return s.div(denom);
}

Tensor Tensor::std(bool unbiased) const {
    return var(unbiased).sqrt();
}

Tensor Tensor::all() const {
    Tensor self_c = to_contiguous(*this);
    std::size_t n = self_c.numel();
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    bool res_bool = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (get_value(self_c, i) == 0.0) {
            res_bool = false;
            break;
        }
    }
    Tensor res = empty({}, DType::Bool);
    res.data<bool>()[0] = res_bool;
    return res;
}

Tensor Tensor::any() const {
    Tensor self_c = to_contiguous(*this);
    std::size_t n = self_c.numel();
    auto get_value = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    bool res_bool = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (get_value(self_c, i) != 0.0) {
            res_bool = true;
            break;
        }
    }
    Tensor res = empty({}, DType::Bool);
    res.data<bool>()[0] = res_bool;
    return res;
}