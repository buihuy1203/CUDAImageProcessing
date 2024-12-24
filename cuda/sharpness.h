#ifndef SHARPNESS_H
#define SHARPNESS_H

void ParallelSharpCUDA(uchar *inputColor, uchar *inputGray uchar *output, int rows, int cols, float sharp_var);

#endif