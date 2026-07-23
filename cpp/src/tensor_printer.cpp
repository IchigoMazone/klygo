#include "klygo/tensor_printer.h"

#include <iomanip>
#include <string>
#include <type_traits>

#include "klygo/dtype.h"
#include "klygo/tensor.h"

namespace
{

template<typename T>
void print_value(std::ostream& os, const T& v)
{
    if constexpr (std::is_floating_point_v<T>)
        os << std::fixed << std::setprecision(4) << v;
    else
        os << v;
}

template<typename T>
void print_recursive(
    std::ostream& os,
    const T* data,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& stride,
    std::size_t dim,
    std::size_t offset,
    std::size_t base_indent   // độ dài của "tensor("
)
{
    // Scalar (tensor 0 chiều)
    if (shape.empty())
    {
        print_value(os, data[offset]);
        return;
    }

    // Chiều cuối cùng: in phẳng 1 hàng
    if (dim == shape.size() - 1)
    {
        os << "[";

        for (std::size_t i = 0; i < shape[dim]; ++i)
        {
            print_value(os, data[offset + i * stride[dim]]);

            if (i + 1 != shape[dim])
                os << ", ";
        }

        os << "]";
        return;
    }

    const std::size_t ndim      = shape.size();
    const std::size_t remaining = ndim - dim; // số chiều còn lại tính cả chiều hiện tại

    os << "[";

    for (std::size_t i = 0; i < shape[dim]; ++i)
    {
        if (i != 0)
        {
            os << ",";

            // Chỉ chèn dòng trống khi vẫn còn > 2 chiều phía dưới
            // (tức không phải cặp 2 chiều cuối cùng - giống cách numpy/pytorch in)
            if (remaining > 2)
                os << "\n";

            os << "\n" << std::string(base_indent + dim + 1, ' ');
        }

        print_recursive(
            os,
            data,
            shape,
            stride,
            dim + 1,
            offset + i * stride[dim],
            base_indent
        );
    }

    os << "]";
}

}

std::ostream& operator<<(
    std::ostream& os,
    const Tensor& tensor
)
{
    static const std::string prefix = "tensor(";
    os << prefix;

    switch (tensor.dtype())
    {
        case DType::Float32:
            print_recursive(
                os, tensor.data<float>(), tensor.shape(), tensor.stride(),
                0, tensor.offset(), prefix.size()
            );
            break;

        case DType::Float64:
            print_recursive(
                os, tensor.data<double>(), tensor.shape(), tensor.stride(),
                0, tensor.offset(), prefix.size()
            );
            break;

        case DType::Int32:
            print_recursive(
                os, tensor.data<int32_t>(), tensor.shape(), tensor.stride(),
                0, tensor.offset(), prefix.size()
            );
            break;

        case DType::Int64:
            print_recursive(
                os, tensor.data<int64_t>(), tensor.shape(), tensor.stride(),
                0, tensor.offset(), prefix.size()
            );
            break;

        case DType::Bool:
            print_recursive(
                os, tensor.data<bool>(), tensor.shape(), tensor.stride(),
                0, tensor.offset(), prefix.size()
            );
            break;
    }

    os << ")";

    return os;
}