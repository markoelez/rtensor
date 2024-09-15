import subprocess
import numpy as np


def rust_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_str = f"{','.join(map(str, a.shape))}:{','.join(map(str, a.flatten()))}"
    b_str = f"{','.join(map(str, b.shape))}:{','.join(map(str, b.flatten()))}"
    
    result = subprocess.run(["./target/release/rtensor", a_str, b_str], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Rust program failed: {result.stderr}")
    
    output = result.stdout.strip()
    shape_str, data_str = output.split(':')
    shape = list(map(int, shape_str.split(',')))
    data = list(map(int, data_str.split(',')))
    
    return np.array(data).reshape(shape)

def test_matmul(a: np.ndarray, b: np.ndarray):
    rust_result = rust_matmul(a, b)
    numpy_result = np.matmul(a, b)

    print(rust_result)
    print(numpy_result)
    
    if np.allclose(rust_result, numpy_result):
        print("Test passed!")
    else:
        print("Test failed!")
        print("Rust result:", rust_result)
        print("NumPy result:", numpy_result)

# Test cases
def run_tests():
    # Basic matrix multiplication
    a = np.ones((2, 3), dtype=int)
    b = np.ones((3, 4), dtype=int)
    test_matmul(a, b)

    # Matrix multiplication with broadcasting
    a = np.ones((1, 2, 1, 3), dtype=int)
    b = np.ones((4, 2, 3, 4), dtype=int)
    test_matmul(a, b)

    # # Large matrix multiplication with non-one values
    a = np.arange(2*3*4*5).reshape((2, 3, 4, 5))
    b = np.arange(3*5*6).reshape((1, 3, 5, 6))
    test_matmul(a, b)

if __name__ == "__main__":
    run_tests()