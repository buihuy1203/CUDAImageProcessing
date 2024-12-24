#ifndef SHARPNESS_H
#define SHARPNESS_H

void ParallelSharpCUDA(unsigned char *inputColor, unsigned char *inputGray, unsigned char *output, int rows, int cols, float sharp_var);

#endif