static __device__ int clamp(int value, int low, int high) {
    return fmaxf(low, fminf(value, high));
}

__global__ void sharpenKernel(const unsigned char *input, unsigned char *output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int kernel2D[3][3] = {
            {0, 1, 0},
            {1, -4, 1},
            {0, 1, 0}
        };
    if (y > 0 && y < rows - 1 && x > 0 && x < cols - 1) {
        int sum = 0;
        for (int k = -1; k <= 1; ++k) {
            for (int l = -1; l <= 1; ++l) {
                if ((y + k) >= 0 && (y + k) < rows && (x + l) >= 0 && (x + l) < cols) {
                    sum += kernel2D[k + 1][l + 1] * input[(y + k) * cols + (x + l)];
                }
            }
        }
        output[y * cols + x] = (unsigned char)clamp(sum, 0, 255);
    }
}

__global__ void applysharpenKernel(const unsigned char *input,const unsigned char *result, unsigned char *output, int rows, int cols, float sharp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        for (int c = 0; c < 3; c++) {
            int idx = (y * cols + x) * 3 + c;
            int grayIdx = y * cols + x;
            int newValue = (float)input[idx] - sharp * (float)result[grayIdx];
            output[idx] = (unsigned char)clamp(newValue, 0, 255);
        }
    }
}

void ParallelSharpCUDA(unsigned char *inputColor, unsigned char *inputGray, unsigned char *output, int rows, int cols, float sharp_var) {

    // Input and output data
    size_t dataGray = rows * cols * sizeof(unsigned char);
    size_t dataSize = rows * cols * 3 * sizeof(unsigned char);
    unsigned char *d_inputData, *d_output, *d_inputColor, *d_result;

    // Allocate device memory
    cudaMalloc(&d_inputData, dataGray);
    cudaMalloc(&d_output, dataSize);
    cudaMalloc(&d_inputColor, dataSize);
    cudaMalloc(&d_result, dataGray);

    // Copy input data to device
    cudaMemcpy(d_inputData, inputGray, dataGray, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputColor, inputColor, dataSize, cudaMemcpyHostToDevice);
    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    sharpenKernel<<<gridSize, blockSize>>>(d_inputData, d_result, rows, cols);
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    applysharpenKernel<<<gridSize, blockSize>>>(d_inputColor, d_result, d_output, rows, cols, sharp_var);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy output data back to host
    cudaMemcpy(output, d_output, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_inputData);
    cudaFree(d_output);
    cudaFree(d_result);
    cudaFree(d_inputColor);
}
