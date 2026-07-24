#include <stdexcept>
#include <vector>
#include "klygo/tensor.h"
#include "klygo/tensor_utils.h"

using namespace std;
using namespace klygo_internal;

// Đổi dạng kích thước Tensor (View) - Không copy dữ liệu, chỉ tính lại shape/stride
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
            target_shape.push_back(0); // Vị trí suy luận chiều tự động
        } else if (shape[i] < 0) {
            throw std::runtime_error("Dimension cannot be negative (except -1)");
        } else {
            target_shape.push_back(static_cast<std::size_t>(shape[i]));
            prod *= static_cast<std::size_t>(shape[i]);
        }
    }
    
    // Tự động tính chiều -1 nếu có
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

    // Tính mảng bước nhảy stride mới cho dạng target_shape
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

// Hoán đổi vị trí hai chiều dim0 và dim1 (Transpose)
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

// Transpose ma trận 2D (t())
Tensor Tensor::t() const {
    if (dim() > 2) {
        throw std::runtime_error("t() expects a tensor with <= 2 dimensions");
    }
    if (dim() < 2) {
        return *this;
    }
    return transpose(0, 1);
}

// Loại bỏ tất cả các chiều có kích thước bằng 1 (Squeeze toàn bộ)
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

// Loại bỏ chiều dim_idx chỉ định nếu chiều đó có kích thước bằng 1 (Squeeze theo dim)
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

// Thêm chiều mới có kích thước bằng 1 tại trí dim_idx (Unsqueeze)
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

// Hoán đổi thứ tự toàn bộ chiều theo danh sách mảng dims (Permute)
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

// Làm phẳng Tensor trong khoảng từ start_dim tới end_dim thành 1D (Flatten)
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
