#include <iostream>
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

// Matrix multiplication kernel
template <typename T>
class MatrixMul;

// Function to initialize a matrix with random values
void initializeMatrix(int* matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i * cols + j] = rand() % 10; // Random values between 0 and 9
        }
    }
}

// Function to print a matrix
void printMatrix(const int* matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Matrix dimensions
    const size_t rowsA = 3, colsA = 4;
    const size_t rowsB = 4, colsB = 5;
    const size_t rowsC = rowsA, colsC = colsB;

    // Create matrices A, B, and C
    int* A = new int[rowsA * colsA];
    int* B = new int[rowsB * colsB];
    int* C = new int[rowsC * colsC];

    // Initialize matrices A and B
    initializeMatrix(A, rowsA, colsA);
    initializeMatrix(B, rowsB, colsB);

    // Print the input matrices
    std::cout << "Matrix A:\n";
    printMatrix(A, rowsA, colsA);

    std::cout << "\nMatrix B:\n";
    printMatrix(B, rowsB, colsB);

    // Create a queue for device execution
    sycl::queue myQueue(sycl::default_selector{});

    // Create buffers for matrices A, B, and C
    sycl::buffer<int, 2> bufferA(A, sycl::range<2>(rowsA, colsA));
    sycl::buffer<int, 2> bufferB(B, sycl::range<2>(rowsB, colsB));
    sycl::buffer<int, 2> bufferC(C, sycl::range<2>(rowsC, colsC));

    // Submit a parallel kernel to the queue for matrix multiplication
    myQueue.submit([&](sycl::handler& cgh) {
        auto matA = bufferA.get_access<sycl::access::mode::read>(cgh);
        auto matB = bufferB.get_access<sycl::access::mode::read>(cgh);
        auto matC = bufferC.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<MatrixMul<int>>(
            sycl::range<2>(rowsC, colsC),
            [=](sycl::item<2> item) {
                int sum = 0;
                for (size_t k = 0; k < colsA; ++k) {
                    sum += matA[item[0]][k] * matB[k][item[1]];
                }
                matC[item] = sum;
            });
    });

    // Access the result back on the host
    myQueue.wait_and_throw();
    std::cout << "\nMatrix C (Result of A * B):\n";
    printMatrix(C, rowsC, colsC);

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
