#include "klygo/tensor_factory.h"

#include <memory>
#include <cstring>
#include <random>

#include "klygo/cpu_allocator.h"
#include "klygo/storage.h"
#include "klygo/tensor_impl.h"

Tensor empty(
    const std::vector<std::size_t>& shape,
    DType dtype
)
{
    std::size_t numel = 1;

    for (auto s : shape) {
        numel *= s;
    }

    std::size_t bytes = numel * dtype_size(dtype);
    auto allocator = std::make_shared<CPUAllocator>();
    auto storage = std::make_shared<Storage>(bytes, allocator);
    auto impl = std::make_shared<TensorImpl>(storage, shape, dtype);

    return Tensor(impl);
}

namespace {

template<typename T>
void fill_data(void* ptr, std::size_t numel, T value) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (std::size_t i = 0; i < numel; ++i) {
        typed_ptr[i] = value;
    }
}

void fill_tensor(Tensor& t, double value) {
    void* ptr = t.data();
    std::size_t numel = t.numel();
    switch (t.dtype()) {
        case DType::Float32:
            fill_data<float>(ptr, numel, static_cast<float>(value));
            break;
        case DType::Float64:
            fill_data<double>(ptr, numel, value);
            break;
        case DType::Int32:
            fill_data<int32_t>(ptr, numel, static_cast<int32_t>(value));
            break;
        case DType::Int64:
            fill_data<int64_t>(ptr, numel, static_cast<int64_t>(value));
            break;
        case DType::Bool:
            fill_data<bool>(ptr, numel, static_cast<bool>(value));
            break;
    }
}

} // namespace

Tensor zeros(
    const std::vector<std::size_t>& shape,
    DType dtype
)
{
    Tensor t = empty(shape, dtype);
    std::size_t bytes = t.numel() * dtype_size(t.dtype());
    if (bytes > 0 && t.data() != nullptr) {
        std::memset(t.data(), 0, bytes);
    }
    return t;
}

Tensor ones(
    const std::vector<std::size_t>& shape,
    DType dtype
)
{
    Tensor t = empty(shape, dtype);
    fill_tensor(t, 1.0);
    return t;
}

Tensor full(
    const std::vector<std::size_t>& shape,
    double value,
    DType dtype
)
{
    Tensor t = empty(shape, dtype);
    fill_tensor(t, value);
    return t;
}

Tensor eye(
    std::size_t n,
    std::size_t m,
    DType dtype
)
{
    Tensor t = zeros({n, m}, dtype);
    void* ptr = t.data();
    std::size_t min_dim = std::min(n, m);
    switch (dtype) {
        case DType::Float32: {
            float* p = static_cast<float*>(ptr);
            for (std::size_t i = 0; i < min_dim; ++i) {
                p[i * m + i] = 1.0f;
            }
            break;
        }
        case DType::Float64: {
            double* p = static_cast<double*>(ptr);
            for (std::size_t i = 0; i < min_dim; ++i) {
                p[i * m + i] = 1.0;
            }
            break;
        }
        case DType::Int32: {
            int32_t* p = static_cast<int32_t*>(ptr);
            for (std::size_t i = 0; i < min_dim; ++i) {
                p[i * m + i] = 1;
            }
            break;
        }
        case DType::Int64: {
            int64_t* p = static_cast<int64_t*>(ptr);
            for (std::size_t i = 0; i < min_dim; ++i) {
                p[i * m + i] = 1;
            }
            break;
        }
        case DType::Bool: {
            bool* p = static_cast<bool*>(ptr);
            for (std::size_t i = 0; i < min_dim; ++i) {
                p[i * m + i] = true;
            }
            break;
        }
    }
    return t;
}

Tensor arange(
    double start,
    double end,
    double step,
    DType dtype
)
{
    if (step == 0) {
        throw std::invalid_argument("step must be non-zero");
    }
    std::vector<double> values;
    if (step > 0) {
        for (double v = start; v < end; v += step) {
            values.push_back(v);
        }
    } else {
        for (double v = start; v > end; v += step) {
            values.push_back(v);
        }
    }
    std::size_t numel = values.size();
    Tensor t = empty({numel}, dtype);
    void* ptr = t.data();
    switch (dtype) {
        case DType::Float32: {
            float* p = static_cast<float*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) {
                p[i] = static_cast<float>(values[i]);
            }
            break;
        }
        case DType::Float64: {
            double* p = static_cast<double*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) {
                p[i] = values[i];
            }
            break;
        }
        case DType::Int32: {
            int32_t* p = static_cast<int32_t*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) {
                p[i] = static_cast<int32_t>(values[i]);
            }
            break;
        }
        case DType::Int64: {
            int64_t* p = static_cast<int64_t*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) {
                p[i] = static_cast<int64_t>(values[i]);
            }
            break;
        }
        case DType::Bool: {
            bool* p = static_cast<bool*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) {
                p[i] = static_cast<bool>(values[i]);
            }
            break;
        }
    }
    return t;
}

Tensor linspace(double start, double end, std::size_t steps, DType dtype) {
    if (steps == 0) {
        throw std::invalid_argument("number of steps must be greater than 0");
    }
    Tensor t = empty({steps}, dtype);
    void* ptr = t.data();
    double step_val = (steps > 1) ? (end - start) / static_cast<double>(steps - 1) : 0.0;
    
    switch (dtype) {
        case DType::Float32: {
            float* p = static_cast<float*>(ptr);
            for (std::size_t i = 0; i < steps; ++i) {
                p[i] = static_cast<float>(start + i * step_val);
            }
            if (steps > 1) p[steps - 1] = static_cast<float>(end);
            break;
        }
        case DType::Float64: {
            double* p = static_cast<double*>(ptr);
            for (std::size_t i = 0; i < steps; ++i) {
                p[i] = start + i * step_val;
            }
            if (steps > 1) p[steps - 1] = end;
            break;
        }
        case DType::Int32: {
            int32_t* p = static_cast<int32_t*>(ptr);
            for (std::size_t i = 0; i < steps; ++i) {
                p[i] = static_cast<int32_t>(start + i * step_val);
            }
            break;
        }
        case DType::Int64: {
            int64_t* p = static_cast<int64_t*>(ptr);
            for (std::size_t i = 0; i < steps; ++i) {
                p[i] = static_cast<int64_t>(start + i * step_val);
            }
            break;
        }
        case DType::Bool: {
            bool* p = static_cast<bool*>(ptr);
            for (std::size_t i = 0; i < steps; ++i) {
                p[i] = static_cast<bool>(start + i * step_val);
            }
            break;
        }
    }
    return t;
}

static std::mt19937& get_rng() {
    thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

Tensor rand(const std::vector<std::size_t>& shape, DType dtype) {
    Tensor t = empty(shape, dtype);
    std::size_t numel = t.numel();
    void* ptr = t.data();
    auto& rng = get_rng();
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    switch (dtype) {
        case DType::Float32: {
            float* p = static_cast<float*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = static_cast<float>(dist(rng));
            break;
        }
        case DType::Float64: {
            double* p = static_cast<double*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = dist(rng);
            break;
        }
        default:
            throw std::invalid_argument("rand only supports floating point dtypes");
    }
    return t;
}

Tensor randn(const std::vector<std::size_t>& shape, DType dtype) {
    Tensor t = empty(shape, dtype);
    std::size_t numel = t.numel();
    void* ptr = t.data();
    auto& rng = get_rng();
    std::normal_distribution<double> dist(0.0, 1.0);

    switch (dtype) {
        case DType::Float32: {
            float* p = static_cast<float*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = static_cast<float>(dist(rng));
            break;
        }
        case DType::Float64: {
            double* p = static_cast<double*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = dist(rng);
            break;
        }
        default:
            throw std::invalid_argument("randn only supports floating point dtypes");
    }
    return t;
}

Tensor randint(int64_t low, int64_t high, const std::vector<std::size_t>& shape, DType dtype) {
    if (low >= high) {
        throw std::invalid_argument("low must be less than high");
    }
    Tensor t = empty(shape, dtype);
    std::size_t numel = t.numel();
    void* ptr = t.data();
    auto& rng = get_rng();
    std::uniform_int_distribution<int64_t> dist(low, high - 1);

    switch (dtype) {
        case DType::Int32: {
            int32_t* p = static_cast<int32_t*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = static_cast<int32_t>(dist(rng));
            break;
        }
        case DType::Int64: {
            int64_t* p = static_cast<int64_t*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = dist(rng);
            break;
        }
        case DType::Float32: {
            float* p = static_cast<float*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = static_cast<float>(dist(rng));
            break;
        }
        case DType::Float64: {
            double* p = static_cast<double*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = static_cast<double>(dist(rng));
            break;
        }
        case DType::Bool: {
            bool* p = static_cast<bool*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = static_cast<bool>(dist(rng) != 0);
            break;
        }
    }
    return t;
}

Tensor empty_like(const Tensor& input, DType dtype) {
    Tensor res = empty(input.shape(), dtype);
    if (input.device().type() != DeviceType::CPU) {
        res = res.to(input.device());
    }
    return res;
}

Tensor zeros_like(const Tensor& input, DType dtype) {
    Tensor res = zeros(input.shape(), dtype);
    if (input.device().type() != DeviceType::CPU) {
        res = res.to(input.device());
    }
    return res;
}

Tensor ones_like(const Tensor& input, DType dtype) {
    Tensor res = ones(input.shape(), dtype);
    if (input.device().type() != DeviceType::CPU) {
        res = res.to(input.device());
    }
    return res;
}

Tensor full_like(const Tensor& input, double value, DType dtype) {
    Tensor res = full(input.shape(), value, dtype);
    if (input.device().type() != DeviceType::CPU) {
        res = res.to(input.device());
    }
    return res;
}

void manual_seed(uint64_t seed) {
    get_rng().seed(static_cast<unsigned int>(seed));
}

Tensor cat(const std::vector<Tensor>& tensors, int64_t dim_idx) {
    if (tensors.empty()) throw std::runtime_error("cat expects a non-empty list of tensors");
    std::size_t nd = tensors[0].dim();
    if (dim_idx < 0) dim_idx += static_cast<int64_t>(nd);
    if (dim_idx < 0 || dim_idx >= static_cast<int64_t>(nd)) throw std::out_of_range("dim out of range");
    
    std::size_t cat_dim = static_cast<std::size_t>(dim_idx);
    std::vector<std::size_t> out_shape = tensors[0].shape();
    std::size_t total_cat_size = 0;
    
    for (const auto& t : tensors) {
        if (t.dim() != nd) throw std::runtime_error("All tensors must have the same number of dimensions");
        for (std::size_t d = 0; d < nd; ++d) {
            if (d != cat_dim && t.shape()[d] != out_shape[d]) {
                throw std::runtime_error("Sizes of tensors must match except in cat dimension");
            }
        }
        total_cat_size += t.shape()[cat_dim];
    }
    out_shape[cat_dim] = total_cat_size;
    
    Tensor res = empty(out_shape, tensors[0].dtype());
    if (tensors[0].device().type() != DeviceType::CPU) res = res.to(tensors[0].device());
    
    std::size_t current_offset = 0;
    for (const auto& t : tensors) {
        Tensor t_c = t.contiguous();
        std::size_t slice_size = t.shape()[cat_dim];
        std::size_t outer_size = 1;
        for (std::size_t d = 0; d < cat_dim; ++d) outer_size *= out_shape[d];
        std::size_t inner_size = 1;
        for (std::size_t d = cat_dim + 1; d < nd; ++d) inner_size *= out_shape[d];
        
        std::size_t elem_bytes = dtype_size(res.dtype());
        char* dst = static_cast<char*>(res.data());
        const char* src = static_cast<const char*>(t_c.data());
        
        for (std::size_t o = 0; o < outer_size; ++o) {
            std::size_t dst_idx = (o * total_cat_size + current_offset) * inner_size;
            std::size_t src_idx = (o * slice_size) * inner_size;
            std::memcpy(dst + dst_idx * elem_bytes, src + src_idx * elem_bytes, slice_size * inner_size * elem_bytes);
        }
        current_offset += slice_size;
    }
    return res;
}

Tensor stack(const std::vector<Tensor>& tensors, int64_t dim_idx) {
    if (tensors.empty()) throw std::runtime_error("stack expects a non-empty list of tensors");
    std::vector<Tensor> unsqueezed;
    for (const auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim_idx));
    }
    return cat(unsqueezed, dim_idx);
}

std::vector<Tensor> split(const Tensor& tensor, std::size_t split_size, int64_t dim_idx) {
    std::size_t nd = tensor.dim();
    if (dim_idx < 0) dim_idx += static_cast<int64_t>(nd);
    if (dim_idx < 0 || dim_idx >= static_cast<int64_t>(nd)) throw std::out_of_range("dim out of range");
    
    std::size_t target_dim = static_cast<std::size_t>(dim_idx);
    std::size_t dim_len = tensor.shape()[target_dim];
    std::vector<Tensor> result;
    
    std::size_t start = 0;
    Tensor t_c = tensor.contiguous();
    while (start < dim_len) {
        std::size_t current_len = std::min(split_size, dim_len - start);
        std::vector<std::size_t> chunk_shape = t_c.shape();
        chunk_shape[target_dim] = current_len;
        
        Tensor chunk_tensor = empty(chunk_shape, t_c.dtype());
        if (t_c.device().type() != DeviceType::CPU) chunk_tensor = chunk_tensor.to(t_c.device());
        
        std::size_t outer_size = 1;
        for (std::size_t d = 0; d < target_dim; ++d) outer_size *= t_c.shape()[d];
        std::size_t inner_size = 1;
        for (std::size_t d = target_dim + 1; d < nd; ++d) inner_size *= t_c.shape()[d];
        
        std::size_t elem_bytes = dtype_size(t_c.dtype());
        char* dst = static_cast<char*>(chunk_tensor.data());
        const char* src = static_cast<const char*>(t_c.data());
        
        for (std::size_t o = 0; o < outer_size; ++o) {
            std::size_t dst_idx = (o * current_len) * inner_size;
            std::size_t src_idx = (o * dim_len + start) * inner_size;
            std::memcpy(dst + dst_idx * elem_bytes, src + src_idx * elem_bytes, current_len * inner_size * elem_bytes);
        }
        result.push_back(chunk_tensor);
        start += current_len;
    }
    return result;
}

std::vector<Tensor> chunk(const Tensor& tensor, std::size_t chunks, int64_t dim_idx) {
    if (chunks == 0) throw std::invalid_argument("chunk expects number of chunks > 0");
    std::size_t nd = tensor.dim();
    if (dim_idx < 0) dim_idx += static_cast<int64_t>(nd);
    std::size_t target_dim = static_cast<std::size_t>(dim_idx);
    std::size_t dim_len = tensor.shape()[target_dim];
    std::size_t split_size = (dim_len + chunks - 1) / chunks;
    return split(tensor, split_size, dim_idx);
}

Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other) {
    if (condition.shape() != input.shape() || input.shape() != other.shape()) {
        throw std::runtime_error("shapes of condition, input, and other must match");
    }
    Tensor cond_c = condition.contiguous();
    Tensor in_c = input.contiguous();
    Tensor oth_c = other.contiguous();
    
    Tensor res = empty(input.shape(), input.dtype());
    std::size_t n = res.numel();
    
    auto get_cond = [](const Tensor& t, std::size_t i) -> bool {
        switch (t.dtype()) {
            case DType::Bool: return t.data<bool>()[i];
            default: return static_cast<double>(t.data<float>()[i]) != 0.0;
        }
    };
    
    auto get_val = [](const Tensor& t, std::size_t i) -> double {
        switch (t.dtype()) {
            case DType::Float32: return static_cast<double>(t.data<float>()[i]);
            case DType::Float64: return t.data<double>()[i];
            case DType::Int32: return static_cast<double>(t.data<int32_t>()[i]);
            case DType::Int64: return static_cast<double>(t.data<int64_t>()[i]);
            case DType::Bool: return static_cast<double>(t.data<bool>()[i]);
            default: return 0.0;
        }
    };
    
    auto set_val = [](Tensor& t, std::size_t i, double val) {
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
        if (get_cond(cond_c, i)) {
            set_val(res, i, get_val(in_c, i));
        } else {
            set_val(res, i, get_val(oth_c, i));
        }
    }
    return res;
}