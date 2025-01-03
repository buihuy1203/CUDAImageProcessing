static float BrightTime=0.0f;

static __device__ int clamp(int value, int low, int high) {
    return fmaxf(low, fminf(value, high));
}

__global__ void brightnessKernel(const unsigned char *input, unsigned char *output, int rows, int cols, int bright) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols) {
        int idx = (y * cols + x) * 3;
        output[idx]= (unsigned char)clamp(input[idx]+(float)bright,0.0f,255.0f); // Blue
        output[idx + 1] = (unsigned char)clamp(input[idx+1]+(float)bright,0.0f,255.0f); // Green
        output[idx + 2] = (unsigned char)clamp(input[idx+2]+(float)bright,0.0f,255.0f); // Red
    }
}

__host__ void ParallelBrightnessCUDA(unsigned char *input,unsigned char *output, int rows, int cols, int bright) {

    // Input and output data
    size_t dataSize = rows * cols * 3 * sizeof(unsigned char);
    unsigned char *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output, dataSize);

    // Create Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy input data to device
    cudaMemcpy(d_input, input, dataSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);

    // Record start time
    cudaEventRecord(start); 

    // Launch kernel
    brightnessKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols, bright);

    // Record stop time
    cudaEventRecord(stop);

    // Wait for GPU to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    BrightTime += milliseconds;

    // Copy output data back to host
    cudaMemcpy(output, d_output, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__host__ float getBrightTime() {
    return BrightTime;
}