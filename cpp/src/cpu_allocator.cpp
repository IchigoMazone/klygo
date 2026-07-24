#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include "klygo/cpu_allocator.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#endif

constexpr std::size_t ALIGNMENT_BYTES = 64; // 64-byte alignment for AVX-512 & AVX2 SIMD

void* CPUAllocator::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
    ptr = _aligned_malloc(bytes, ALIGNMENT_BYTES);
    if (!ptr) throw std::bad_alloc();
#else
    if (posix_memalign(&ptr, ALIGNMENT_BYTES, bytes) != 0) {
        throw std::bad_alloc();
    }
#endif
    return ptr;
}

void CPUAllocator::deallocate(void* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}