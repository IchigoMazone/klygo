#include <iostream>
#include <memory>
#include <vector>

#include "klygo/cpu_allocator.h"
#include "klygo/storage.h"
#include "klygo/tensor_impl.h"
#include "klygo/tensor.h"
#include "klygo/dtype.h"

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

    std::cout << "========== data<T>() ==========\n\n";

    std::cout << "void*   : " << x.data() << '\n';

    float* p = x.data<float>();

    std::cout << "float*  : " << p << '\n';

    std::cout << "\nWrite data...\n";

    for (std::size_t i = 0; i < x.numel(); i++) {
        p[i] = static_cast<float>(i + 1);
    }

    std::cout << "\nRead data:\n";

    for (std::size_t i = 0; i < x.numel(); i++) {
        std::cout << p[i] << " ";
    }

    std::cout << '\n';

    return 0;
}