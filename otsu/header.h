#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
void getHistogram(cv::Mat img, int *histogram);
int otsu(int *histogram, int histoSize);
void binarize(cv::Mat img, int threshold);