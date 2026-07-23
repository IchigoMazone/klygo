#include "klygo/storage.h"

using namespace std;

Storage::Storage(size_t bytes, shared_ptr<Allocator> allocator) : bytes_(bytes), allocator_(std::move(allocator)) {
    data_ = allocator_->allocate(bytes_);
}

Storage::~Storage() {
    if (data_) {
        allocator_->deallocate(data_);
    }
}

void* Storage::data() const {
    return data_;
}

size_t Storage::bytes() const {
    return bytes_;
}