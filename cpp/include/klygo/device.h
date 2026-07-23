#pragma once
#include <string>
#include <stdexcept>
#include <algorithm>

enum class DeviceType {
    CPU,
    CUDA
};

class Device {
public:
    Device(DeviceType type) : type_(type) {}

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

    DeviceType type() const { return type_; }

    std::string toString() const {
        return (type_ == DeviceType::CPU) ? "cpu" : "cuda";
    }

    bool operator==(const Device& other) const {
        return type_ == other.type_;
    }

    bool operator!=(const Device& other) const {
        return type_ != other.type_;
    }

private:
    DeviceType type_;
};
