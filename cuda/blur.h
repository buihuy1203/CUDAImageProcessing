#ifndef BLUR_H
#define BLUR_H


float* createGaussianKernel(int size, float sigma);
void ParallelBlurCUDA(unsigned char *input,unsigned char *output,int rows, int cols, float blur_sar);

#endif