#pragma once

#include <ostream>

class Tensor;

std::ostream& operator<<(
    std::ostream& os,
    const Tensor& tensor
);