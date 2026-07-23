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

// Đổ dữ liệu có cả số âm - hay dùng để test dấu trừ có làm lệch cột hay không
template<typename T>
void fill_signed(Tensor& t)
{
    T* p = t.data<T>();
    long v = 1;

    for (std::size_t i = 0; i < t.numel(); ++i)
    {
        p[i] = static_cast<T>(v);
        v = (i % 3 == 1) ? -v - 1 : v + 1; // xen kẽ dương/âm
    }
}

void fill_bool(Tensor& t)
{
    bool* p = t.data<bool>();

    for (std::size_t i = 0; i < t.numel(); ++i)
        p[i] = (i % 2 == 0);
}

int main()
{
    std::cout << "========== 0D (scalar) ==========\n\n";
    Tensor t0 = empty({}, DType::Float32);
    fill<float>(t0);
    std::cout << t0 << "\n\n";

    std::cout << "========== 1D ==========\n\n";
    Tensor t1 = empty({6}, DType::Float32);
    fill<float>(t1);
    std::cout << t1 << "\n\n";

    std::cout << "========== 2D ==========\n\n";
    Tensor t2 = empty({2, 3}, DType::Float32);
    fill<float>(t2);
    std::cout << t2 << "\n\n";

    std::cout << "========== 3D ==========\n\n";
    Tensor t3 = empty({2, 2, 3}, DType::Float32);
    fill<float>(t3);
    std::cout << t3 << "\n\n";

    std::cout << "========== 4D ==========\n\n";
    Tensor t4 = empty({2, 2, 2, 2}, DType::Float32);
    fill<float>(t4);
    std::cout << t4 << "\n\n";

    std::cout << "========== 5D ==========\n\n";
    Tensor t5 = empty({2, 2, 2, 2, 2}, DType::Float32);
    fill<float>(t5);
    std::cout << t5 << "\n\n";

    std::cout << "========== 6D ==========\n\n";
    Tensor t6 = empty({2, 1, 2, 1, 2, 2}, DType::Float32);
    fill<float>(t6);
    std::cout << t6 << "\n\n";

    std::cout << "========== Float64 ==========\n\n";
    Tensor td = empty({2, 2, 3}, DType::Float64);
    fill<double>(td);
    std::cout << td << "\n\n";

    std::cout << "========== Int32 ==========\n\n";
    Tensor ti32 = empty({2, 3}, DType::Int32);
    fill<int32_t>(ti32);
    std::cout << ti32 << "\n\n";

    std::cout << "========== Int64 (có số âm) ==========\n\n";
    Tensor ti64 = empty({2, 2, 3}, DType::Int64);
    fill_signed<int64_t>(ti64);
    std::cout << ti64 << "\n\n";

    std::cout << "========== Bool ==========\n\n";
    Tensor tb = empty({2, 4}, DType::Bool);
    fill_bool(tb);
    std::cout << tb << "\n\n";

    std::cout << "========== Cạnh biên: chiều = 1 ==========\n\n";
    Tensor tedge = empty({1, 1, 3}, DType::Float32);
    fill<float>(tedge);
    std::cout << tedge << "\n\n";

    std::cout << "========== Cạnh biên: rỗng (0 phần tử) ==========\n\n";
    Tensor tempty = empty({0, 3}, DType::Float32);
    std::cout << tempty << "\n\n";

    return 0;
}