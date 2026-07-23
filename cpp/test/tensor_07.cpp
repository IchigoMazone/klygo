#include <iostream>

#include "klygo/tensor_factory.h"
#include "klygo/tensor_printer.h"

template<typename T>
void fill(Tensor& t)
{
    T* p = t.data<T>();

    for (std::size_t i = 0; i < t.numel(); ++i)
        p[i] = static_cast<T>(i + 1);
}

int main()
{
    std::cout << "========== 1D ==========\n\n";

    Tensor t1 = empty({6}, DType::Float32);
    fill<float>(t1);

    std::cout << t1 << "\n\n";


    std::cout << "========== 2D ==========\n\n";

    Tensor t2 = empty({2,3}, DType::Float32);
    fill<float>(t2);

    std::cout << t2 << "\n\n";


    std::cout << "========== 3D ==========\n\n";

    Tensor t3 = empty({2,2,3}, DType::Float32);
    fill<float>(t3);

    std::cout << t3 << "\n\n";


    std::cout << "========== 4D ==========\n\n";

    Tensor t4 = empty({2,2,2,2}, DType::Float32);
    fill<float>(t4);

    std::cout << t4 << "\n\n";

    return 0;
}