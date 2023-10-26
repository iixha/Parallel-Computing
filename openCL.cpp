#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <CL/cl.h>
#include "matrix.h"

using namespace std;

void FillMatricesRandomly(Matrix<double> &A, Matrix<double> &B);
void PrintMatrices(Matrix<double> &A, Matrix<double> &B, Matrix<double> &C);

int randomHigh = 100;
int randomLow = 0;

int main(int argc, char *argv[]) {
    cout << "Starting a parallel matrix multiplication using OpenCL." << endl;

    if (argv[1] == NULL) {
        cout << "ERROR: The program must be executed in the following way  \n\n  \t \"./a N \"  \n\n where N is an integer. \n \n " << endl;
        return 1;
    }

    int N = atoi(argv[1]);
    cout << "The matrices are: " << N << "x" << N << endl;

    int numberOfRowsA = N;
    int numberOfColsA = N;
    int numberOfRowsB = N;
    int numberOfColsB = N;

    Matrix<double> A = Matrix<double>(numberOfRowsA, numberOfColsA);
    Matrix<double> B = Matrix<double>(numberOfRowsB, numberOfColsB);
    Matrix<double> C = Matrix<double>(numberOfRowsA, numberOfColsB);

    FillMatricesRandomly(A, B);

    cl_uint ret_num_platforms;
    clGetPlatformIDs(0, NULL, &ret_num_platforms);
    if (ret_num_platforms == 0) {
        cout << "No OpenCL platforms found." << endl;
        return 1;
    }

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_uint ret_num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, NULL, &ret_num_devices);
    if (ret_num_devices == 0) {
        cout << "No OpenCL GPU devices found." << endl;
        return 1;
    }

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, NULL);

    size_t size = N;
    Matrix<double> A_host = A;
    Matrix<double> B_host = B;
    Matrix<double> C_host(size, size);

    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * size * sizeof(double), A_host.data(), NULL);
    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * size * sizeof(double), B_host.data(), NULL);
    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * size * sizeof(double), NULL, NULL);

    const char* kernel_source =
        "__kernel void matrix_mul(__global double* A, __global double* B, __global double* C, int N) {\n"
        "    int i = get_global_id(0);\n"
        "    int j = get_global_id(1);\n"
        "    double sum = 0;\n"
        "    for (int k = 0; k < N; k++) {\n"
        "        sum += A[i * N + k] * B[k * N + j];\n"
        "    }\n"
        "    C[i * N + j] = sum;\n"
        "}\n";

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "matrix_mul", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C);
    clSetKernelArg(kernel, 3, sizeof(int), &size);

    size_t global_work_size[2] = { size, size };
    size_t local_work_size[2] = { 1, 1 };

    clock_t start = clock();
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    clFinish(command_queue);
    clock_t end = clock();
    double matrixCalculationTime = (double)(end - start) / CLOCKS_PER_SEC;

    clEnqueueReadBuffer(command_queue, buffer_C, CL_TRUE, 0, size * size * sizeof(double), C_host.data(), 0, NULL, NULL);

    cout << "\nTotal multiplication time = " << matrixCalculationTime << " seconds" << endl;

    PrintMatrices(A, B, C_host);

    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}

void FillMatricesRandomly(Matrix<double> &A, Matrix<double> &B) {
    srand(time(NULL));

    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            A(i, j) = rand() % (randomHigh - randomLow) + randomLow;
        }
    }

    for (int i = 0; i < B.rows(); i++) {
        for (int j = 0; j < B.cols(); j++) {
            B(i, j) = rand() % (randomHigh - randomLow) + randomLow;
        }
    }
}

void PrintMatrices(Matrix<double> &A, Matrix<double> &B, Matrix<double> &C) {
    // Same as your original function
}
