#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include "cuda\blur.h"
#include "cuda\brightness.h"
#include "cuda\saturation.h"
#include "cuda\sharpness.h"
#include "cuda\ycrcb.h"
using namespace std;
using namespace cv;

int main() {
    Mat image1 = imread("meo_xe_tang (26).jpg", IMREAD_COLOR);
    if (image1.empty()) {
            cout << "Can't read image" << endl;
            return -1;
        }
    imshow("Original", image1);
    int rows = image1.rows;
    int cols = image1.cols;

    Mat YCrCBIm(image1.size(), CV_8UC3);
    Mat SatIm(image1.size(), CV_8UC3);
    Mat SharpIm(image1.size(), CV_8UC3);
    Mat BrightIm(image1.size(), CV_8UC3);
    Mat BlurIm(image1.size(), CV_8UC3);
    Mat grayImage;
    cvtColor(image1, grayImage, COLOR_BGR2GRAY);

    vector<uchar> inputColor(rows * cols * 3);
    vector<uchar> inputGray(rows * cols);
    vector<uchar> outputYCrCB(rows * cols * 3);
    vector<uchar> outputSat(rows * cols * 3);
    vector<uchar> outputSharp(rows * cols * 3);
    vector<uchar> outputBright(rows * cols * 3);
    vector<uchar> outputBlur(rows * cols * 3);

    memcpy(inputColor.data(), image1.data, rows * cols * 3);
    memcpy(inputGray.data(), grayImage.data, rows * cols);

    auto start = chrono::high_resolution_clock::now();
    ParallelYCrCBCUDA(inputColor.data(), outputYCrCB.data(), rows, cols);
    ParallelSatCUDA(inputColor.data(), outputSat.data(), rows, cols, 1);
    ParallelSharpCUDA(inputColor.data(), inputGray.data(), outputSharp.data(), rows, cols, 5);
    ParallelBrightnessCUDA(inputColor.data(), outputBright.data(), rows, cols, 1);
    ParallelBlurCUDA(inputColor.data(), outputBlur.data(), rows, cols, -100);

    auto end = chrono::high_resolution_clock::now();

    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    memcpy(YCrCBIm.data, outputYCrCB.data(), rows * cols * 3);
    memcpy(SatIm.data, outputSat.data(), rows * cols * 3);
    memcpy(SharpIm.data, outputSharp.data(), rows * cols * 3);
    memcpy(BrightIm.data, outputBright.data(), rows * cols * 3);
    memcpy(BlurIm.data, outputBlur.data(), rows * cols * 3);

    imshow("YCrCB", YCrCBIm);
    imshow("Saturation", SatIm);
    imshow("Sharpness", SharpIm);
    imshow("Brightness", BrightIm);
    imshow("Blur", BlurIm);
    waitKey(0);
    
    return 0;
}