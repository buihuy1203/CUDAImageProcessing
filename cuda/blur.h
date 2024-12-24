#ifndef BLUR_H
#define BLUR_H

#define M_PI 3.14159

float* createGaussianKernel(int size, float sigma);
void ParallelBlurCUDA(uchar *input,uchar *output,int rows, int cols, float blur_sar);

#endif