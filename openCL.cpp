#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <CL/cl.h>
#include "matrix.h"

#define SIZE 512 // Adjust this based on your matrix size
#define TILE_SIZE 16

void FillMatricesRandomly(Matrix<double> &A, Matrix<double> &B);
void PrintMatrices(Matrix<double> &A, Matrix<double> &B, Matrix<double> &C);

int main() {
    // Create matrices and fill them randomly
    Matrix<double> A(SIZE, SIZE);
    Matrix<double> B(SIZE, SIZE);
    Matrix<double> C(SIZE, SIZE);

    FillMatricesRandomly(A, B);

    // Load OpenCL source code from a file
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("matrix_mul.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    source_str = (char *)malloc(SIZE);
    source_size = fread(source_str, 1, SIZE, fp);
    fclose(fp);

    // OpenCL initialization
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * SIZE * sizeof(double), NULL, NULL);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * SIZE * sizeof(double), NULL, NULL);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * SIZE * sizeof(double), NULL, NULL);

    clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, SIZE * SIZE * sizeof(double), A.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, SIZE * SIZE * sizeof(double), B.data(), 0, NULL, NULL);

    // Compile OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create OpenCL kernels
    cl_kernel kernel = clCreateKernel(program, "matrix_mul", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

    // Set global and local work size
    size_t local_item_size[2] = { TILE_SIZE, TILE_SIZE };
    size_t global_item_size[2] = { SIZE, SIZE };

    // Execute OpenCL kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);
    clFinish(command_queue);

    // Read the result back from the GPU
    clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, SIZE * SIZE * sizeof(double), C.data(), 0, NULL, NULL);

    // Cleanup OpenCL resources
    clReleaseMemObject(a_mem_obj);
    clReleaseMemObject(b_mem_obj);
    clReleaseMemObject(c_mem_obj);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    // Print matrices and cleanup
    PrintMatrices(A, B, C);
    free(source_str);

    return 0;
}

void FillMatricesRandomly(Matrix<double> &A, Matrix<double> &B) {
    srand(time(NULL));
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            A(i, j) = static_cast<double>(rand() % 100) / 10.0; // Random values between 0 and 9.9
            B(i, j) = static_cast<double>(rand() % 100) / 10.0;
        }
    }
}

void PrintMatrices(Matrix<double> &A, Matrix<double> &B, Matrix<double> &C) {
    cout << "\n\nMatrix A" << endl;
  for (int i = 0; i < A.rows(); i++) {
    cout << endl << endl;
    for (int j = 0; j < A.cols(); j++)
      cout << A(i,j) << " ";
  }
  
  cout << "\n\n\n\nMatrix B" << endl;  
  
  for (int i = 0; i < B.rows(); i++) {
    cout << "\n" << endl;
    for (int j = 0; j < B.cols(); j++)
      cout << B(i,j) << " ";
  }
  
  cout << "\n\n\n\nMultiplied Matrix C" << endl;  
  
  for (int i = 0; i < C.rows(); i++) {
    cout << "\n" << endl;  
    for (int j = 0; j < C.cols(); j++)
      cout << C(i,j) << " ";
  }
  
  cout << endl << endl << endl;  
}
}
