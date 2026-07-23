#include <iostream>

#include "klygo/tensor_factory.h"

int main()
{
    Tensor x = empty({2, 3}, DType::Float32);

    std::cout << "numel   : " << x.numel() << '\n';
    std::cout << "dim     : " << x.dim() << '\n';

    std::cout << "shape   : ";
    for (auto s : x.shape()) {
        std::cout << s << " ";
    }

    std::cout << '\n';

    std::cout << "stride  : ";
    for (auto s : x.stride()) {
        std::cout << s << " ";
    }

    std::cout << '\n';
    std::cout << "dtype size : " << dtype_size(x.dtype()) << '\n';
    std::cout << "storage : " << x.storage().get() << '\n';
    std::cout << "data    : " << x.data() << '\n';
    std::cout << "\nWrite data...\n";

    float* p = x.data<float>();

    for (std::size_t i = 0; i < x.numel(); i++)
        p[i] = static_cast<float>(i + 1);

    std::cout << "\nRead data:\n";

    for (std::size_t i = 0; i < x.numel(); i++)
        std::cout << p[i] << " ";

    std::cout << '\n';

    return 0;
}