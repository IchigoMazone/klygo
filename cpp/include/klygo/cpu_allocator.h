#pragma once
#include <cstddef>
#include "klygo/allocator.h"

class CPUAllocator : public Allocator {

public:
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    
};