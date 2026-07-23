#include <cstddef>
#include "klygo/cpu_allocator.h"

void* CPUAllocator::allocate(size_t bytes) {
    return ::operator new(bytes);
}

void CPUAllocator::deallocate(void* ptr) {
    ::operator delete(ptr);
}