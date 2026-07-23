#include <iostream>

#include "klygo/tensor_factory.h"
#include "klygo/tensor_printer.h"

int main()
{
    {
        Tensor x = empty({2, 3}, DType::Float32);

        float* p = x.data<float>();

        for (int i = 0; i < x.numel(); i++)
            p[i] = 999.0f;

        std::cout << "x:\n";
        std::cout << x << "\n\n";
    } // x bị hủy

    Tensor y = empty({2, 3}, DType::Float32);

    std::cout << "y:\n";
    std::cout << y << "\n";
}