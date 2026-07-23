#pragma once
#include <cstddef>
#include <memory>
#include "klygo/allocator.h"

class Storage {

public:
    Storage(std::size_t bytes, std::shared_ptr<Allocator> allocator);
    ~Storage();
    void* data() const;
    std::size_t bytes() const;

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    std::shared_ptr<Allocator> allocator_;

};