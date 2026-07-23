#include <iostream>
#include <memory>

#include "klygo/tensor_factory.h"
#include "klygo/tensor_printer.h"

int main() {

    Tensor x = empty({2, 3}, DType::Float32);
    float* p = x.data<float>();

    // for (int i = 0; i < x.numel(); ++i) {
    //     p[i] = static_cast<float>(i + 1);
    // }

    std::cout << "Tensor:\n";
    std::cout << x << "\n\n";

    std::cout << "Shape : ";
    for (auto s : x.shape())
        std::cout << s << " ";

    std::cout << "\n";

    std::cout << "Stride: ";
    for (auto s : x.stride())
        std::cout << s << " ";

    std::cout << "\n";

    return 0;
}