
__device__ int clamp(int value, int low, int high) {
    return fmaxf(low, fminf(value, high));
}

__global__ void brightnessKernel(const uchar *input, uchar *output, int rows, int cols, int bright) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols) {
        int idx = (y * cols + x) * 3;
        output[idx]= (uchar)clamp(input[idx]+(float)bright,0.0f,255.0f); // Blue
        output[idx + 1] = (uchar)clamp(input[idx+1]+(float)bright,0.0f,255.0f); // Green
        output[idx + 2] = (uchar)clamp(input[idx+2]+(float)bright,0.0f,255.0f); // Red
    }
}

void ParallelBrightnessCUDA(uchar *input,uchar *output, int rows, int cols, int bright) {

    // Input and output data
    size_t dataSize = rows * cols * 3 * sizeof(uchar);
    uchar *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output, dataSize);

    // Copy input data to device
    cudaMemcpy(d_input, input, dataSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    brightnessKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols, bright);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy output data back to host
    cudaMemcpy(output, d_output, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
