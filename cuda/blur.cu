
__device__ int clamp(int value, int low, int high) {
    return fmaxf(low, fminf(value, high));
}

__global__ void blurKernel(const uchar *input, uchar *output,const float *kernel1D  int rows, int cols, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfSize = kernelSize/2;

    if (x < rows && y < cols) {
        float sum[3] = {0.0f, 0.0f, 0.0f};

        for (int kx = -halfSize; kx <= halfSize; ++kx) {
            for (int ky = -halfSize; ky <= halfSize; ++ky) {
                int px = clamp(x + kx, 0, rows - 1);
                int py = clamp(y + ky, 0, cols - 1);

                int pixelIndex = (px * cols + py) * channels;
                float weight = kernel1D[(kx + halfSize) * kernelSize + (ky + halfSize)];

                sum[0] += (float)input[pixelIndex] * weight;       // Blue channel
                sum[1] += (float)input[pixelIndex + 1] * weight;   // Green channel
                sum[2] += (float)input[pixelIndex + 2] * weight;   // Red channel
            }
        }

        int outputIndex = (x * cols + y) * channels;
        output[outputIndex] = (uchar)clamp(sum[0], 0.0f, 255.0f);
        output[outputIndex + 1] = (uchar)clamp(sum[1], 0.0f, 255.0f);
        output[outputIndex + 2] = (uchar)clamp(sum[2], 0.0f, 255.0f);
    }
}

float* createGaussianKernel(int size, float sigma) {
    float* kernel = new float[size * size]; // Cấp phát bộ nhớ cho mảng 1D
    float sum = 0.0f;
    int halfSize = size / 2;

    // Tạo kernel Gaussian
    for (int x = -halfSize; x <= halfSize; ++x) {
        for (int y = -halfSize; y <= halfSize; ++y) {
            int idx = (x + halfSize) * size + (y + halfSize); // Biến đổi chỉ số 2D thành 1D
            kernel[idx] = (1.0f / (2.0f * M_PI * sigma * sigma)) * expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            sum += kernel[idx];
        }
    }

    // Chuẩn hóa kernel
    for (int i = 0; i < size * size; ++i) {
        kernel[i] /= sum;
    }

    return kernel; // Trả về con trỏ mảng
}

void ParallelBlurCUDA(uchar *input,uchar *output,int rows, int cols, float blur_sar) {
    // Gaussian kernel
    int kernelSize = 7;
    // Input and output data
    size_t dataSize = rows * cols * 3 * sizeof(uchar);
    uchar *d_input, *d_output;
    float *d_kernel;
    float *kernel1D = createGaussianKernel(kernelSize, blur_sar);
    // Allocate device memory
    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output, dataSize);
    cudaMalloc(&d_kernel, 7 * 7 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel1D, 7 * 7 * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    blurKernel<<<gridSize, blockSize>>>(d_input, d_output,d_kernel, rows, cols, kernelSize);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy output data back to host
    cudaMemcpy(output, d_output, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

}
