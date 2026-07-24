#pragma once
#include <string>
#include <stdexcept>
#include <algorithm>

/**
 * @brief Enum định nghĩa loại thiết bị phần cứng thực thi (CPU hoặc CUDA GPU).
 */
enum class DeviceType {
    CPU,   // Bộ xử lý trung tâm CPU
    CUDA   // Card đồ họa NVIDIA GPU (CUDA)
};

/**
 * @brief Lớp biểu diễn thiết bị phần cứng thực tính toán cho Tensor.
 * Tương đương với torch.device("cpu") hoặc torch.device("cuda") trong PyTorch.
 */
class Device {
public:
    /**
     * @brief Khởi tạo Device từ enum DeviceType.
     * @param type Loại thiết bị (DeviceType::CPU hoặc DeviceType::CUDA)
     */
    Device(DeviceType type) : type_(type) {}

    /**
     * @brief Khởi tạo Device từ chuỗi văn bản (ví dụ: "cpu", "cuda").
     * @param device_str Chuỗi tên thiết bị
     */
    Device(const std::string& device_str) {
        std::string lower_str = device_str;
        std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
        if (lower_str == "cpu") {
            type_ = DeviceType::CPU;
        } else if (lower_str == "cuda") {
            type_ = DeviceType::CUDA;
        } else {
            throw std::invalid_argument("Unknown device type: " + device_str);
        }
    }

    /// Lấy loại thiết bị dưới dạng enum DeviceType
    DeviceType type() const { return type_; }

    /// Chuyển đổi thông tin thiết bị sang chuỗi văn bản ("cpu" hoặc "cuda")
    std::string toString() const {
        return (type_ == DeviceType::CPU) ? "cpu" : "cuda";
    }

    /// So sánh hai thiết bị có cùng loại hay không
    bool operator==(const Device& other) const {
        return type_ == other.type_;
    }

    /// So sánh hai thiết bị có khác loại hay không
    bool operator!=(const Device& other) const {
        return type_ != other.type_;
    }

private:
    DeviceType type_; // Loại thiết bị nội bộ
};
