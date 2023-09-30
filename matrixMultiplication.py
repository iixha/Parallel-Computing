import random
import time

# Function to generate a random matrix of size rows x cols
def generate_random_matrix(rows, cols):
    return [[random.randint(1, 10) for _ in range(cols)] for _ in range(rows)]

# Function to perform matrix multiplication
def matrix_multiply(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Matrix dimensions are not compatible for multiplication")

    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

# Define the size of the matrices for big data (adjust these values as needed)
matrix_A_rows = 500  # Increase this value for larger data
matrix_A_cols = 500  # Increase this value for larger data
matrix_B_rows = 500  # Increase this value for larger data
matrix_B_cols = 500  # Increase this value for larger data

# Generate random matrices
matrix_A = generate_random_matrix(matrix_A_rows, matrix_A_cols)
matrix_B = generate_random_matrix(matrix_B_rows, matrix_B_cols)

# Record the start time
start_time = time.time()

# Multiply the matrices
result_matrix = matrix_multiply(matrix_A, matrix_B)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Matrix multiplication took {elapsed_time:.2f} seconds for {matrix_A_rows}x{matrix_A_cols} and {matrix_B_rows}x{matrix_B_cols} matrices.")
for row in result_matrix:
    print(row)