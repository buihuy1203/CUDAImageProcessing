#ifndef BRIGHTNESS_H
#define BRIGHTNESS_H

void ParallelBrightnessCUDA(unsigned char *input,unsigned char *output,int rows, int cols, int bright);
float getBrightTime();

#endif