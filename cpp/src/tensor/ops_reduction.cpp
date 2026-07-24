#include <algorithm>
#include <stdexcept>
#include <vector>
#include "klygo/tensor.h"
#include "klygo/tensor_utils.h"

using namespace std;
using namespace klygo_internal;

// Phép cộng tổng mảng trên CPU với phân luồng OpenMP reduction
template<typename T>
double sum_typed(const T* ptr, std::size_t n) {
    double total = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:total)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        total += static_cast<double>(ptr[i]);
    }
    return total;
}

// === Tính tổng toàn bộ các phần tử (Sum All) ===
Tensor Tensor::sum() const {
    Tensor self_c = to_contiguous(*this);
    std::size_t n = self_c.numel();
    double total = 0.0;
    switch (self_c.dtype()) {
        case DType::Float32: total = sum_typed(self_c.data<float>(), n); break;
        case DType::Float64: total = sum_typed(self_c.data<double>(), n); break;
        case DType::Int32:   total = sum_typed(self_c.data<int32_t>(), n); break;
        case DType::Int64:   total = sum_typed(self_c.data<int64_t>(), n); break;
        case DType::Bool:    total = sum_typed(self_c.data<bool>(), n); break;
    }
    
    DType res_dtype = (dtype() == DType::Bool) ? DType::Int64 : dtype();
    Tensor res = empty({}, res_dtype);
    switch (res_dtype) {
        case DType::Float32: res.data<float>()[0] = static_cast<float>(total); break;
        case DType::Float64: res.data<double>()[0] = total; break;
        case DType::Int32:   res.data<int32_t>()[0] = static_cast<int32_t>(total); break;
        case DType::Int64:   res.data<int64_t>()[0] = static_cast<int64_t>(total); break;
        case DType::Bool:    res.data<bool>()[0] = static_cast<bool>(total); break;
    }
    return res;
}

// === Tính tổng theo một chiều dim chỉ định (Sum Dim) ===
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
            case DType::Int32:   return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64:   return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool:    return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    
    auto add_value = [](Tensor& t, std::size_t i, double val) {
        switch (t.dtype()) {
            case DType::Float32: t.data<float>()[i] += static_cast<float>(val); break;
            case DType::Float64: t.data<double>()[i] += val; break;
            case DType::Int32:   t.data<int32_t>()[i] += static_cast<int32_t>(val); break;
            case DType::Int64:   t.data<int64_t>()[i] += static_cast<int64_t>(val); break;
            case DType::Bool:    t.data<bool>()[i] = static_cast<bool>(static_cast<double>(t.data<bool>()[i]) + val); break;
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

// === Giá trị trung bình toàn bộ phần tử (Mean All) ===
Tensor Tensor::mean() const {
    Tensor s = sum();
    return s.div(static_cast<double>(numel()));
}

// === Giá trị trung bình theo chiều dim (Mean Dim) ===
Tensor Tensor::mean(int64_t dim, bool keepdim) const {
    Tensor s = sum(dim, keepdim);
    std::size_t ndim = this->dim();
    if (dim < 0) dim += ndim;
    return s.div(static_cast<double>(size(dim)));
}

// Phép tích mảng trên CPU
template<typename T>
double prod_typed(const T* ptr, std::size_t n) {
    double p = 1.0;
    for (std::size_t i = 0; i < n; ++i) p *= static_cast<double>(ptr[i]);
    return p;
}

// === Tích toàn bộ phần tử (Prod) ===
Tensor Tensor::prod() const {
    Tensor self_c = to_contiguous(*this);
    double p = 1.0;
    std::size_t n = self_c.numel();
    switch (self_c.dtype()) {
        case DType::Float32: p = prod_typed(self_c.data<float>(), n); break;
        case DType::Float64: p = prod_typed(self_c.data<double>(), n); break;
        case DType::Int32:   p = prod_typed(self_c.data<int32_t>(), n); break;
        case DType::Int64:   p = prod_typed(self_c.data<int64_t>(), n); break;
        case DType::Bool:    p = prod_typed(self_c.data<bool>(), n); break;
    }
    Tensor res = empty({}, dtype());
    switch (dtype()) {
        case DType::Float32: res.data<float>()[0] = static_cast<float>(p); break;
        case DType::Float64: res.data<double>()[0] = p; break;
        case DType::Int32:   res.data<int32_t>()[0] = static_cast<int32_t>(p); break;
        case DType::Int64:   res.data<int64_t>()[0] = static_cast<int64_t>(p); break;
        case DType::Bool:    res.data<bool>()[0] = static_cast<bool>(p); break;
    }
    return res;
}

// Phép tìm giá trị lớn nhất trên CPU
template<typename T>
double max_typed(const T* ptr, std::size_t n) {
    T mx = ptr[0];
    for (std::size_t i = 1; i < n; ++i) if (ptr[i] > mx) mx = ptr[i];
    return static_cast<double>(mx);
}

// === Giá trị lớn nhất toàn phần tử (Max All) ===
Tensor Tensor::max() const {
    Tensor self_c = to_contiguous(*this);
    if (numel() == 0) throw std::runtime_error("cannot perform max on empty tensor");
    double mx = 0.0;
    std::size_t n = self_c.numel();
    switch (self_c.dtype()) {
        case DType::Float32: mx = max_typed(self_c.data<float>(), n); break;
        case DType::Float64: mx = max_typed(self_c.data<double>(), n); break;
        case DType::Int32:   mx = max_typed(self_c.data<int32_t>(), n); break;
        case DType::Int64:   mx = max_typed(self_c.data<int64_t>(), n); break;
        case DType::Bool:    mx = max_typed(self_c.data<bool>(), n); break;
    }
    Tensor res = empty({}, dtype());
    switch (dtype()) {
        case DType::Float32: res.data<float>()[0] = static_cast<float>(mx); break;
        case DType::Float64: res.data<double>()[0] = mx; break;
        case DType::Int32:   res.data<int32_t>()[0] = static_cast<int32_t>(mx); break;
        case DType::Int64:   res.data<int64_t>()[0] = static_cast<int64_t>(mx); break;
        case DType::Bool:    res.data<bool>()[0] = static_cast<bool>(mx); break;
    }
    return res;
}

// Phép tìm giá trị nhỏ nhất trên CPU
template<typename T>
double min_typed(const T* ptr, std::size_t n) {
    T mn = ptr[0];
    for (std::size_t i = 1; i < n; ++i) if (ptr[i] < mn) mn = ptr[i];
    return static_cast<double>(mn);
}

// === Giá trị nhỏ nhất toàn phần tử (Min All) ===
Tensor Tensor::min() const {
    Tensor self_c = to_contiguous(*this);
    if (numel() == 0) throw std::runtime_error("cannot perform min on empty tensor");
    double mn = 0.0;
    std::size_t n = self_c.numel();
    switch (self_c.dtype()) {
        case DType::Float32: mn = min_typed(self_c.data<float>(), n); break;
        case DType::Float64: mn = min_typed(self_c.data<double>(), n); break;
        case DType::Int32:   mn = min_typed(self_c.data<int32_t>(), n); break;
        case DType::Int64:   mn = min_typed(self_c.data<int64_t>(), n); break;
        case DType::Bool:    mn = min_typed(self_c.data<bool>(), n); break;
    }
    Tensor res = empty({}, dtype());
    switch (dtype()) {
        case DType::Float32: res.data<float>()[0] = static_cast<float>(mn); break;
        case DType::Float64: res.data<double>()[0] = mn; break;
        case DType::Int32:   res.data<int32_t>()[0] = static_cast<int32_t>(mn); break;
        case DType::Int64:   res.data<int64_t>()[0] = static_cast<int64_t>(mn); break;
        case DType::Bool:    res.data<bool>()[0] = static_cast<bool>(mn); break;
    }
    return res;
}

// === Phép tính phương sai (Variance - Var) ===
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

// === Phép tính độ lệch chuẩn (Standard Deviation - Std) ===
Tensor Tensor::std(bool unbiased) const {
    return var(unbiased).sqrt();
}

template<typename T>
bool all_typed(const T* ptr, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) if (ptr[i] == static_cast<T>(0)) return false;
    return true;
}

// === Phép kiểm tra tất cả phần tử đều đúng/khác 0 (All) ===
Tensor Tensor::all() const {
    Tensor self_c = to_contiguous(*this);
    std::size_t n = self_c.numel();
    bool res_bool = true;
    switch (self_c.dtype()) {
        case DType::Float32: res_bool = all_typed(self_c.data<float>(), n); break;
        case DType::Float64: res_bool = all_typed(self_c.data<double>(), n); break;
        case DType::Int32:   res_bool = all_typed(self_c.data<int32_t>(), n); break;
        case DType::Int64:   res_bool = all_typed(self_c.data<int64_t>(), n); break;
        case DType::Bool:    res_bool = all_typed(self_c.data<bool>(), n); break;
    }
    Tensor res = empty({}, DType::Bool);
    res.data<bool>()[0] = res_bool;
    return res;
}

template<typename T>
bool any_typed(const T* ptr, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) if (ptr[i] != static_cast<T>(0)) return true;
    return false;
}

// === Phép kiểm tra có ít nhất một phần tử đúng/khác 0 (Any) ===
Tensor Tensor::any() const {
    Tensor self_c = to_contiguous(*this);
    std::size_t n = self_c.numel();
    bool res_bool = false;
    switch (self_c.dtype()) {
        case DType::Float32: res_bool = any_typed(self_c.data<float>(), n); break;
        case DType::Float64: res_bool = any_typed(self_c.data<double>(), n); break;
        case DType::Int32:   res_bool = any_typed(self_c.data<int32_t>(), n); break;
        case DType::Int64:   res_bool = any_typed(self_c.data<int64_t>(), n); break;
        case DType::Bool:    res_bool = any_typed(self_c.data<bool>(), n); break;
    }
    Tensor res = empty({}, DType::Bool);
    res.data<bool>()[0] = res_bool;
    return res;
}
