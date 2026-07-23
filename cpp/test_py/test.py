# import os
# os.add_dll_directory(r"C:\msys64\ucrt64\bin")

import klygo


def line():
    print("-" * 50)



print("=" * 50)
print("       KLYGO PYTHON BINDING TEST")
print("=" * 50)



# =========================
# Test Module
# =========================

line()

print("[1] Module")

print("Module loaded:")
print(klygo)

print("Available:")
print(dir(klygo))



# =========================
# Test DType
# =========================

line()

print("[2] dtype")

print("float32:")
print(klygo.float32)

print("float64:")
print(klygo.float64)

print("int32:")
print(klygo.int32)

print("int64:")
print(klygo.int64)

print("bool:")
print(klygo.bool)



# =========================
# Test Tensor class
# =========================

line()

print("[3] Tensor class")

print(klygo.Tensor)



# =========================
# Create Tensor
# =========================

line()

print("[4] Tensor instance")


# Hiện tại cần hàm C++ tạo tensor
# ví dụ:
#
# x = klygo.test_tensor()
#
# Sau này thay bằng:
#
# x = klygo.empty(
#       [2,3],
#       klygo.DType.Float32
# )


try:
    print("Testing empty([2, 3], float32):")
    x = klygo.empty([2, 3], klygo.float32)
    print("Tensor created")
    print(f"numel: {x.numel()}")
    print(f"shape: {x.shape}")
    print(f"stride: {x.stride()}")
    print(f"offset: {x.offset()}")
    print(f"dtype: {x.dtype}")
    print(f"data pointer: {hex(x.data_ptr())}")

    print("\nTesting zeros([2, 2], int32):")
    z = klygo.zeros([2, 2], klygo.int32)
    print(f"shape: {z.shape}, dtype: {z.dtype}, numel: {z.numel()}")

    print("\nTesting ones([3, 1], float64):")
    o = klygo.ones([3, 1], klygo.float64)
    print(f"shape: {o.shape}, dtype: {o.dtype}, numel: {o.numel()}")

    print("\nTesting full([2, 4], 42.0, int64):")
    f = klygo.full([2, 4], 42.0, klygo.int64)
    print(f"shape: {f.shape}, dtype: {f.dtype}, numel: {f.numel()}")

except Exception as e:
    print(f"Error testing APIs: {e}")



line()

print("TEST FINISHED")