__global__ void yCrCBKernel(const uchar *input, uchar *output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols) {
        int idx = (y * cols + x) * 3;
        // Input Value
        uchar blue = input[idx];
        uchar green = input[idx + 1];
        uchar red = input[idx + 2];

        // RGB to YCrCB
        uchar Y = (uchar)(0.299f * red + 0.587f * green + 0.114f * blue);
        uchar Cb = (uchar)(128.0f + (blue - Y) * 0.564f);
        uchar Cr = (uchar)(128.0f + (red - Y) * 0.713f);

        // Output Value
        output[idx]     = Y;
        output[idx + 1] = Cb;
        output[idx + 2] = Cr;
    }
}

void ParallelYCrCBCUDA(uchar *input, uchar *output, int rows, int cols) {

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
    yCrCBKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy output data back to host
    cudaMemcpy(output, d_output, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
