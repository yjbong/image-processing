#include "header.h"

void getHistogram(cv::Mat img, int *histogram) {
	for (int i = 0; i < img.size().height; i++)
		for (int j = 0; j < img.size().width; j++)
			histogram[img.at<unsigned char>(i, j)]++;
}

int otsu(int *histogram, int histoSize) {
	double bestScore = -1.0;
	int threshold = -1;

	for (int i = 0; i < histoSize; i++) {

		// Calculate left group size
		int leftSize = 0;
		for (int j = 0; j <= i; j++) leftSize += histogram[j];
		// Calculate left group mean
		double leftMean = 0.0;
		for (int j = 0; j <= i; j++) leftMean += j * (((double)histogram[j]) / leftSize);
		// Calculate left group deviation
		double leftDev = 0.0;
		for (int j = 0; j <= i; j++) leftDev += (j - leftMean)*(j - leftMean)*(((double)histogram[j]) / leftSize);

		// Calculate right group size
		int rightSize = 0;
		for (int j = i + 1; j < histoSize; j++) rightSize += histogram[j];
		// Calculate right group mean
		double rightMean = 0.0;
		for (int j = i + 1; j < histoSize; j++) rightMean += j * (((double)histogram[j]) / rightSize);
		// Calculate right group deviation
		double rightDev = 0.0;
		for (int j = i + 1; j < histoSize; j++) rightDev += (j - rightMean)*(j - rightMean)*(((double)histogram[j]) / rightSize);

		// Calculate score of current threshold i
		double score = leftSize * leftDev + rightSize * rightDev;
		if (bestScore < 0 || bestScore > score) {
			bestScore = score;
			threshold = i;
		}
	}

	return threshold;
}

void binarize(cv::Mat img, int threshold) {
	for (int i = 0; i < img.size().height; i++)
		for (int j = 0; j < img.size().width; j++)
			img.at<unsigned char>(i, j) = (img.at<unsigned char>(i, j) >= threshold ? 255 : 0);
}