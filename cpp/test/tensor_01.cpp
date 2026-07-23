#include <iostream>
#include <memory>
#include "klygo/tensor.h"
#include "klygo/tensor_impl.h"
#include "klygo/storage.h"
#include "klygo/cpu_allocator.h"
#include "klygo/dtype.h"


int main() {

    auto allocator = std::make_shared<CPUAllocator>();
    auto storage = std::make_shared<Storage>(
        2 * 3 * sizeof(float),
        allocator
    );

    auto impl = std::make_shared<TensorImpl>(
        storage,
        std::vector<std::size_t>{2,3},
        DType::Float32
    );

    Tensor x(impl);

    std::cout << "numel = " << x.numel() << std::endl;
    std::cout << "address = " << x.data() << std::endl;
    return 0;
}