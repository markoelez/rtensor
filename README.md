# rtensor

This project implements a simple NDArray structure in Rust with matrix multiplication functionality. It includes a Python test suite that compares the Rust implementation's results with NumPy's results.

## Prerequisites

- Rust (with Cargo)
- Python 3.7+
- NumPy

## Running the Test Suite

1. Ensure you're in the project root directory.

2. Build the Rust binary:
   ```
   cargo build --release
   ```

3. Run the Python test script:
   ```
   python run_tests.py
   ```

   This script will run several test cases, comparing the results of the Rust implementation with NumPy's results.
