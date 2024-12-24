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
    auto start = chrono::high_resolution_clock::now();
    for(int i = 1; i <=200; i++){
    string path = "imagetest/meo_xe_tang (" + to_string(i) + ").jpg";
    Mat image1 = imread(path, IMREAD_COLOR);
    if (image1.empty()) {
            cout << "Can't read image" << endl;
            return -1;
        }
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

    ParallelYCrCBCUDA(inputColor.data(), outputYCrCB.data(), rows, cols);
    ParallelSatCUDA(inputColor.data(), outputSat.data(), rows, cols, 1);
    ParallelSharpCUDA(inputColor.data(), inputGray.data(), outputSharp.data(), rows, cols, 5);
    ParallelBrightnessCUDA(inputColor.data(), outputBright.data(), rows, cols, 1);
    ParallelBlurCUDA(inputColor.data(), outputBlur.data(), rows, cols, -100);

    memcpy(YCrCBIm.data, outputYCrCB.data(), rows * cols * 3);
    memcpy(SatIm.data, outputSat.data(), rows * cols * 3);
    memcpy(SharpIm.data, outputSharp.data(), rows * cols * 3);
    memcpy(BrightIm.data, outputBright.data(), rows * cols * 3);
    memcpy(BlurIm.data, outputBlur.data(), rows * cols * 3);
    
    cout << "Image " << i << " finish"<<endl;
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "CUDA Time Total: " << duration.count() << "s" << endl;
    cout << "CUDA Time Kernel: "<< getBlurTime() + getBrightTime() + getSatTime() + getSharpTime() + getYCrCBTime() << "ms" << endl;

    //Brightness CUDA
    auto startbright = chrono::high_resolution_clock::now();
    for(int i = 1; i <=200; i++){
    string path = "imagetest/meo_xe_tang (" + to_string(i) + ").jpg";
    Mat image1 = imread(path, IMREAD_COLOR);
    if (image1.empty()) {
            cout << "Can't read image" << endl;
            return -1;
        }
    int rows = image1.rows;
    int cols = image1.cols;

    Mat BrightIm(image1.size(), CV_8UC3);

    vector<uchar> inputColor(rows * cols * 3);
    vector<uchar> outputBright(rows * cols * 3);

    memcpy(inputColor.data(), image1.data, rows * cols * 3);

    ParallelBrightnessCUDA(inputColor.data(), outputBright.data(), rows, cols, 1);

    memcpy(BrightIm.data, outputBright.data(), rows * cols * 3);
    
    cout << "Image " << i << " finish"<<endl;
    }
    auto endbright = chrono::high_resolution_clock::now();
    chrono::duration<double> durationbright = endbright - startbright;

    cout << "CUDA Time Bright Total: " << durationbright.count() << "s" << endl;
    cout << "CUDA Time Bright Kernel: "<< getBrightTime()<< " ms" << endl;

    //Saturation CUDA
    auto startsat = chrono::high_resolution_clock::now();
    for(int i = 1; i <=200; i++){
    string path = "imagetest/meo_xe_tang (" + to_string(i) + ").jpg";
    Mat image1 = imread(path, IMREAD_COLOR);
    if (image1.empty()) {
            cout << "Can't read image" << endl;
            return -1;
        }
    int rows = image1.rows;
    int cols = image1.cols;

    Mat SatIm(image1.size(), CV_8UC3);

    vector<uchar> inputColor(rows * cols * 3);
    vector<uchar> outputSat(rows * cols * 3);

    memcpy(inputColor.data(), image1.data, rows * cols * 3);

    ParallelSatCUDA(inputColor.data(), outputSat.data(), rows, cols, 1);

    memcpy(SatIm.data, outputSat.data(), rows * cols * 3);
    
    cout << "Image " << i << " finish"<<endl;
    }
    auto endsat = chrono::high_resolution_clock::now();
    chrono::duration<double> durationsat = endsat - startsat;

    cout << "CUDA Time Saturation Total: " << durationsat.count() << "s" << endl;
    cout << "CUDA Time Saturation Kernel: "<< getSatTime()<< " ms" << endl;

    //Sharpness CUDA
    auto startsharp = chrono::high_resolution_clock::now();
    for(int i = 1; i <=200; i++){
    string path = "imagetest/meo_xe_tang (" + to_string(i) + ").jpg";
    Mat image1 = imread(path, IMREAD_COLOR);
    if (image1.empty()) {
            cout << "Can't read image" << endl;
            return -1;
        }
    int rows = image1.rows;
    int cols = image1.cols;

    Mat SharpIm(image1.size(), CV_8UC3);
    Mat grayImage;
    cvtColor(image1, grayImage, COLOR_BGR2GRAY);

    vector<uchar> inputColor(rows * cols * 3);
    vector<uchar> inputGray(rows * cols);
    vector<uchar> outputSharp(rows * cols * 3);

    memcpy(inputColor.data(), image1.data, rows * cols * 3);
    memcpy(inputGray.data(), grayImage.data, rows * cols);

    ParallelSharpCUDA(inputColor.data(), inputGray.data(), outputSharp.data(), rows, cols, 5);

    memcpy(SharpIm.data, outputSharp.data(), rows * cols * 3);

    cout << "Image " << i << " finish"<<endl;
    }
    auto endsharp = chrono::high_resolution_clock::now();
    chrono::duration<double> durationsharp = endsharp - startsharp;

    cout << "CUDA Time Sharpness Total: " << durationsharp.count() << "s" << endl;
    cout << "CUDA Time Sharpness Kernel: "<< getSharpTime()<< " ms" << endl;

    //YCrCb CUDA
    auto startycrcb = chrono::high_resolution_clock::now();
    for(int i = 1; i <=200; i++){
    string path = "imagetest/meo_xe_tang (" + to_string(i) + ").jpg";
    Mat image1 = imread(path, IMREAD_COLOR);
    if (image1.empty()) {
            cout << "Can't read image" << endl;
            return -1;
        }
    int rows = image1.rows;
    int cols = image1.cols;

    Mat YCrCBIm(image1.size(), CV_8UC3);

    vector<uchar> inputColor(rows * cols * 3);
    vector<uchar> outputYCrCB(rows * cols * 3);

    memcpy(inputColor.data(), image1.data, rows * cols * 3);

    ParallelYCrCBCUDA(inputColor.data(), outputYCrCB.data(), rows, cols);

    memcpy(YCrCBIm.data, outputYCrCB.data(), rows * cols * 3);

    cout << "Image " << i << " finish"<<endl;
    }

    auto endycrcb = chrono::high_resolution_clock::now();
    chrono::duration<double> durationycrcb = endycrcb - startycrcb;

    cout << "CUDA Time YCrCb Total: " << durationycrcb.count() << "s" << endl;
    cout << "CUDA Time YCrCb Kernel: "<< getYCrCBTime()<< " ms" << endl;

    //Blur CUDA
    auto startblur = chrono::high_resolution_clock::now();
    for(int i = 1; i <=200; i++){
    string path = "imagetest/meo_xe_tang (" + to_string(i) + ").jpg";
    Mat image1 = imread(path, IMREAD_COLOR);
    if (image1.empty()) {
            cout << "Can't read image" << endl;
            return -1;
        }
    int rows = image1.rows;
    int cols = image1.cols;

    Mat BlurIm(image1.size(), CV_8UC3);

    vector<uchar> inputColor(rows * cols * 3);
    vector<uchar> outputBlur(rows * cols * 3);

    memcpy(inputColor.data(), image1.data, rows * cols * 3);

    ParallelBlurCUDA(inputColor.data(), outputBlur.data(), rows, cols, -100);

    memcpy(BlurIm.data, outputBlur.data(), rows * cols * 3);   

    cout << "Image " << i << " finish"<<endl;
    }
    auto endblur = chrono::high_resolution_clock::now();
    chrono::duration<double> durationblur = endblur - startblur;

    cout << "CUDA Time Blur Total: " << durationblur.count() << "s" << endl;
    cout << "CUDA Time Blur Kernel: "<< getBlurTime()<< " ms" << endl;
    
    return 0;
}