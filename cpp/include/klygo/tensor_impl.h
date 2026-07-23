#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include "klygo/storage.h"
#include "klygo/dtype.h"
#include "klygo/device.h"

class TensorImpl {

public:
    TensorImpl(
        std::shared_ptr<Storage> storage, 
        const std::vector<std::size_t>& shape,
        DType dtype,
        Device device = Device(DeviceType::CPU)
    );

    TensorImpl(
        std::shared_ptr<Storage> storage, 
        const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& stride,
        std::size_t offset,
        DType dtype,
        Device device = Device(DeviceType::CPU)
    );

    const std::vector<std::size_t>& shape() const;
    const std::vector<std::size_t>& stride() const;
    
    std::size_t offset() const;
    std::size_t numel() const;
    std::size_t dim() const;
    std::size_t size(std::size_t dim) const;
    DType dtype() const;
    Device device() const;

    void* data() const;
    std::shared_ptr<Storage> storage() const;

private:
    void compute_stride();

private:
    std::shared_ptr<Storage> storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> stride_;
    std::size_t offset_ = 0;
    DType dtype_;
    Device device_;
};