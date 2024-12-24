__global__ void yCrCBKernel(const unsigned char *input, unsigned char *output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols) {
        int idx = (y * cols + x) * 3;
        // Input Value
        unsigned char blue = input[idx];
        unsigned char green = input[idx + 1];
        unsigned char red = input[idx + 2];

        // RGB to YCrCB
        unsigned char Y = (unsigned char)(0.299f * red + 0.587f * green + 0.114f * blue);
        unsigned char Cb = (unsigned char)(128.0f + (blue - Y) * 0.564f);
        unsigned char Cr = (unsigned char)(128.0f + (red - Y) * 0.713f);

        // Output Value
        output[idx]     = Y;
        output[idx + 1] = Cb;
        output[idx + 2] = Cr;
    }
}

__host__ void ParallelYCrCBCUDA(unsigned char *input, unsigned char *output, int rows, int cols) {

    // Input and output data
    size_t dataSize = rows * cols * 3 * sizeof(unsigned char);
    unsigned char *d_input, *d_output;

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
    yCrCBKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy output data back to host
    cudaMemcpy(output, d_output, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
