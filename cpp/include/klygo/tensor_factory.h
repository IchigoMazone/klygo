#pragma once

#include <vector>

#include "klygo/tensor.h"
#include "klygo/dtype.h"

Tensor empty(
    const std::vector<std::size_t>& shape,
    DType dtype
);

Tensor zeros(
    const std::vector<std::size_t>& shape,
    DType dtype
);

Tensor ones(
    const std::vector<std::size_t>& shape,
    DType dtype
);

Tensor full(
    const std::vector<std::size_t>& shape,
    double value,
    DType dtype
);

Tensor eye(
    std::size_t n,
    std::size_t m,
    DType dtype
);

Tensor arange(
    double start,
    double end,
    double step,
    DType dtype
);

Tensor linspace(
    double start,
    double end,
    std::size_t steps,
    DType dtype
);

Tensor rand(
    const std::vector<std::size_t>& shape,
    DType dtype
);

Tensor randn(
    const std::vector<std::size_t>& shape,
    DType dtype
);

Tensor randint(
    int64_t low,
    int64_t high,
    const std::vector<std::size_t>& shape,
    DType dtype
);

Tensor empty_like(
    const Tensor& input,
    DType dtype
);

Tensor zeros_like(
    const Tensor& input,
    DType dtype
);

Tensor ones_like(
    const Tensor& input,
    DType dtype
);

Tensor full_like(
    const Tensor& input,
    double value,
    DType dtype
);

void manual_seed(uint64_t seed);

Tensor cat(const std::vector<Tensor>& tensors, int64_t dim = 0);
Tensor stack(const std::vector<Tensor>& tensors, int64_t dim = 0);
std::vector<Tensor> split(const Tensor& tensor, std::size_t split_size, int64_t dim = 0);
std::vector<Tensor> chunk(const Tensor& tensor, std::size_t chunks, int64_t dim = 0);
Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other);