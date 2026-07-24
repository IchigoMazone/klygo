#include <cmath>
#include <functional>
#include <algorithm>
#include "klygo/tensor.h"
#include "klygo/tensor_utils.h"

using namespace std;
using namespace klygo_internal;

// === Phép lũy thừa theo số mũ Scalar (Pow) ===
Tensor Tensor::pow(double exponent) const {
    return unary_op_dispatch_cuda(*this, [exponent](double x) { return std::pow(x, exponent); }, [exponent](const void* a, void* c, std::size_t n, DType dtype) {
        launch_pow_scalar_cuda(a, exponent, c, n, dtype);
    });
}

// === Phép lũy thừa theo Tensor khác (Pow Tensor) ===
Tensor Tensor::pow(const Tensor& exponent) const {
    return binary_op_dispatch(*this, exponent, [](double x, double y) { return std::pow(x, y); }, [](const void*, const void*, void*, std::size_t, DType){});
}

// === Phép căn bậc hai (Sqrt) ===
Tensor Tensor::sqrt() const {
    return unary_op_dispatch_cuda(*this, [](double x) { return std::sqrt(x); }, launch_sqrt_cuda);
}

// === Phép hàm mũ e^x (Exp) ===
Tensor Tensor::exp() const {
    return unary_op_dispatch_cuda(*this, [](double x) { return std::exp(x); }, launch_exp_cuda);
}

// === Phép logarithm tự nhiên ln(x) (Log) ===
Tensor Tensor::log() const {
    return unary_op_dispatch_cuda(*this, [](double x) { return std::log(x); }, launch_log_cuda);
}

// === Phép lấy giá trị tuyệt đối |x| (Abs) ===
Tensor Tensor::abs() const {
    return unary_op_dispatch_cuda(*this, [](double x) { return std::abs(x); }, launch_abs_cuda);
}

// === Phép đổi dấu -x (Neg) ===
Tensor Tensor::neg() const {
    return unary_op_dispatch_cuda(*this, [](double x) { return -x; }, launch_neg_cuda);
}

// === Phép cắt dải giá trị [min_val, max_val] (Clamp) ===
Tensor Tensor::clamp(double min_val, double max_val) const {
    return unary_op_dispatch_cuda(*this, [min_val, max_val](double x) {
        return std::max(min_val, std::min(max_val, x));
    }, [min_val, max_val](const void* a, void* c, std::size_t n, DType dtype) {
        launch_clamp_cuda(a, min_val, max_val, c, n, dtype);
    });
}

// === Phép so sánh bằng Tensor == Tensor (Equal) ===
Tensor Tensor::eq(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::equal_to<double>(), launch_eq_cuda);
}

// === Phép so sánh bằng Tensor == Scalar (Equal Scalar) ===
Tensor Tensor::eq(double other) const {
    return scalar_op_dispatch_cuda(*this, other, std::equal_to<double>(), launch_eq_scalar_cuda);
}

// === Phép so sánh khác Tensor != Tensor (Not Equal) ===
Tensor Tensor::ne(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::not_equal_to<double>(), launch_ne_cuda);
}

// === Phép so sánh khác Tensor != Scalar (Not Equal Scalar) ===
Tensor Tensor::ne(double other) const {
    return scalar_op_dispatch_cuda(*this, other, std::not_equal_to<double>(), launch_ne_scalar_cuda);
}

// === Phép so sánh nhỏ hơn Tensor < Tensor (Less Than) ===
Tensor Tensor::lt(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::less<double>(), launch_lt_cuda);
}

// === Phép so sánh nhỏ hơn Tensor < Scalar (Less Than Scalar) ===
Tensor Tensor::lt(double other) const {
    return scalar_op_dispatch_cuda(*this, other, std::less<double>(), launch_lt_scalar_cuda);
}

// === Phép so sánh nhỏ hơn hoặc bằng Tensor <= Tensor (Less Equal) ===
Tensor Tensor::le(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::less_equal<double>(), launch_le_cuda);
}

// === Phép so sánh nhỏ hơn hoặc bằng Tensor <= Scalar (Less Equal Scalar) ===
Tensor Tensor::le(double other) const {
    return scalar_op_dispatch_cuda(*this, other, std::less_equal<double>(), launch_le_scalar_cuda);
}

// === Phép so sánh lớn hơn Tensor > Tensor (Greater Than) ===
Tensor Tensor::gt(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::greater<double>(), launch_gt_cuda);
}

// === Phép so sánh lớn hơn Tensor > Scalar (Greater Than Scalar) ===
Tensor Tensor::gt(double other) const {
    return scalar_op_dispatch_cuda(*this, other, std::greater<double>(), launch_gt_scalar_cuda);
}

// === Phép so sánh lớn hơn hoặc bằng Tensor >= Tensor (Greater Equal) ===
Tensor Tensor::ge(const Tensor& other) const {
    return binary_op_dispatch(*this, other, std::greater_equal<double>(), launch_ge_cuda);
}

// === Phép so sánh lớn hơn hoặc bằng Tensor >= Scalar (Greater Equal Scalar) ===
Tensor Tensor::ge(double other) const {
    return scalar_op_dispatch_cuda(*this, other, std::greater_equal<double>(), launch_ge_scalar_cuda);
}
