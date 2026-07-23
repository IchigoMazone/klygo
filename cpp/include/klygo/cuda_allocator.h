#pragma once
#include <cstddef>
#include <stdexcept>
#include "klygo/allocator.h"

#ifdef KLYGO_USE_CUDA
#include <cuda_runtime.h>
#endif

class CUDAAllocator : public Allocator {
public:
    void* allocate(std::size_t bytes) override {
#ifdef KLYGO_USE_CUDA
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA memory allocation failed: ") + cudaGetErrorString(err));
        }
        return ptr;
#else
        throw std::runtime_error("CUDA is not enabled in this build.");
#endif
    }

    void deallocate(void* ptr) override {
#ifdef KLYGO_USE_CUDA
        if (ptr) {
            cudaFree(ptr);
        }
#else
        throw std::runtime_error("CUDA is not enabled in this build.");
#endif
    }
};
