#include <iostream>
#include <memory>
#include <vector>

#include "klygo/cpu_allocator.h"
#include "klygo/storage.h"
#include "klygo/tensor_impl.h"
#include "klygo/tensor.h"

int main() {

    auto allocator = std::make_shared<CPUAllocator>();

    auto storage = std::make_shared<Storage>(
        6 * sizeof(float),
        allocator
    );

    auto impl = std::make_shared<TensorImpl>(
        storage,
        std::vector<std::size_t>{2, 3},
        DType::Float32
    );

    Tensor x(impl);

    std::cout << "\n=== Tensor ===\n";
    std::cout << "numel   : " << x.numel() << '\n';
    std::cout << "dim     : " << x.dim() << '\n';
    std::cout << "shape   : ";
    for (auto s : x.shape())
        std::cout << s << ' ';
    std::cout << '\n';
    std::cout << "stride  : ";
    for (auto s : x.stride())
        std::cout << s << ' ';
    std::cout << '\n';
    std::cout << "offset  : " << impl->offset() << '\n';
    std::cout << "storage : " << x.storage().get() << '\n';
    std::cout << "data    : " << x.data() << '\n';
    std::cout << "impl    : " << x.impl().get() << '\n';
    return 0;

}