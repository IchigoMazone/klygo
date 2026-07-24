#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include "klygo/storage.h"
#include "klygo/dtype.h"
#include "klygo/device.h"

/**
 * @brief Class chứa metadata và triển khai nội bộ cho Tensor.
 * Quản lý thông tin kích thước (shape), bước nhảy (stride), offset bộ nhớ, kiểu dữ liệu (dtype) và thiết bị (device).
 * Tương đương với c10::TensorImpl trong PyTorch.
 */
class TensorImpl {
public:
    /**
     * @brief Khởi tạo TensorImpl với bước nhảy stride mặc định (tính tự động cho mảng liên tiếp).
     */
    TensorImpl(
        std::shared_ptr<Storage> storage, 
        const std::vector<std::size_t>& shape,
        DType dtype,
        Device device = Device(DeviceType::CPU)
    );

    /**
     * @brief Khởi tạo TensorImpl với bước nhảy stride và offset tùy chỉnh (dùng cho các view/slice).
     */
    TensorImpl(
        std::shared_ptr<Storage> storage, 
        const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& stride,
        std::size_t offset,
        DType dtype,
        Device device = Device(DeviceType::CPU)
    );

    /// Lấy mảng kích thước (shape) của tensor
    const std::vector<std::size_t>& shape() const;

    /// Lấy mảng bước nhảy (stride) của tensor
    const std::vector<std::size_t>& stride() const;
    
    /// Lấy offset vị trí bắt đầu dữ liệu trong Storage
    std::size_t offset() const;

    /// Lấy tổng số phần tử (numel)
    std::size_t numel() const;

    /// Lấy số chiều (ndim)
    std::size_t dim() const;

    /// Lấy kích thước của một chiều chỉ định
    std::size_t size(std::size_t dim) const;

    /// Lấy kiểu dữ liệu DType
    DType dtype() const;

    /// Lấy thiết bị phần cứng (CPU hoặc CUDA)
    Device device() const;

    /// Lấy con trỏ dữ liệu thực tế (đã tính kèm offset)
    void* data() const;

    /// Lấy đối tượng Storage chứa bộ nhớ thô
    std::shared_ptr<Storage> storage() const;

private:
    /// Tính toán mảng stride mặc định dựa trên shape
    void compute_stride();

private:
    std::shared_ptr<Storage> storage_;   // Con trỏ tới bộ nhớ Storage
    std::vector<std::size_t> shape_;     // Kích thước các chiều
    std::vector<std::size_t> stride_;    // Bước nhảy các chiều
    std::size_t offset_ = 0;             // Offset phân đoạn dữ liệu
    DType dtype_;                        // Kiểu dữ liệu
    Device device_;                      // Thiết bị thực thi
};