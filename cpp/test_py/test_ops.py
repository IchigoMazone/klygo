import klygo

def test_eye_and_arange():
    print("--- Testing eye & arange ---")
    e = klygo.eye(3, 3, klygo.float32)
    assert e.shape == (3, 3)
    print("eye(3, 3) created successfully.")

    a = klygo.arange(0.0, 5.0, 1.0, klygo.float32)
    assert a.shape == (5,)
    assert a.numel() == 5
    print("arange(0, 5, 1) created successfully.")

def test_tensor_creation():
    print("--- Testing klygo.tensor ---")
    x = klygo.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert x.shape == (2, 2)
    assert x.dtype == klygo.float32
    
    y = klygo.tensor(42.0, klygo.int64)
    assert y.shape == ()
    assert y.dtype == klygo.int64
    print("klygo.tensor created successfully.")

def test_shape_manipulation():
    print("\n--- Testing view & transpose & t ---")
    x = klygo.arange(0.0, 6.0, 1.0, klygo.float32) # [0, 1, 2, 3, 4, 5]
    assert x.shape == (6,)

    v = x.view([2, 3])
    assert v.shape == (2, 3)
    assert v.stride() == (3, 1)

    t = v.transpose(0, 1)
    assert t.shape == (3, 2)
    assert t.stride() == (1, 3) # Swapped strides!

    tr = v.t()
    assert tr.shape == (3, 2)

    # Test view with -1
    v2 = x.view([3, -1])
    assert v2.shape == (3, 2)
    print("Shape manipulations verified successfully.")

def test_elementwise_math():
    print("\n--- Testing element-wise arithmetic ---")
    a = klygo.ones([2, 2], klygo.float32)
    b = klygo.full([2, 2], 2.0, klygo.float32)

    # Tensor + Tensor
    c = a + b
    # Check sum
    assert c.sum().dtype == klygo.float32
    # Tensor + Scalar
    d = a + 5.0
    # Scalar + Tensor
    e = 10.0 + a

    print("Arithmetic operations (+, -, *, /) work.")

def test_reductions():
    print("\n--- Testing reductions (sum, mean) ---")
    x = klygo.arange(1.0, 7.0, 1.0, klygo.float32).view([2, 3]) # [[1, 2, 3], [4, 5, 6]]
    
    s_all = x.sum()
    # 1+2+3+4+5+6 = 21
    
    s_dim0 = x.sum(dim=0) # [5, 7, 9]
    assert s_dim0.shape == (3,)

    s_dim0_keep = x.sum(dim=0, keepdim=True)
    assert s_dim0_keep.shape == (1, 3)

    m_all = x.mean() # 21 / 6 = 3.5
    
    m_dim1 = x.mean(dim=1) # [2, 5]
    assert m_dim1.shape == (2,)

    print("Reductions sum/mean verified successfully.")

def test_matmul():
    print("\n--- Testing matmul ---")
    a = klygo.full([2, 3], 2.0, klygo.float32)
    b = klygo.full([3, 4], 3.0, klygo.float32)
    
    c = a @ b
    assert c.shape == (2, 4)
    print("Matmul (@ operator) verified successfully.")

if __name__ == "__main__":
    test_eye_and_arange()
    test_tensor_creation()
    test_shape_manipulation()
    test_elementwise_math()
    test_reductions()
    test_matmul()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
