#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void calculateHistogram(const Mat& image, Mat& hist, int histSize);
void stretchHistogram(const Mat& image, Mat& stretchedImage, double minVal, double maxVal);
void equalizeHistogram(const Mat& image, Mat& equalizedImage);

int main(void) {
    // Load the boy image in GrayScale mode
    Mat boyImage = imread("boy.jpg", 0);

    // Check if the boy image is loaded properly
    if (boyImage.empty()) {
        cout << "Cannot load boy image!" << endl;
        return -1;
    }

    // Define a generic 3x3 kernel with values to be defined
    Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -7, 1, 1, 1, 1); // Example filter

    // Create a Mat to store the filtered image
    Mat filteredBoyImage = Mat::zeros(boyImage.size(), boyImage.type());

    // Perform convolution
    for (int y = 1; y < boyImage.rows - 1; y++) {
        for (int x = 1; x < boyImage.cols - 1; x++) {
            float sum = 0.0;
            for (int k = -1; k <= 1; k++) {
                for (int j = -1; j <= 1; j++) {
                    sum += kernel.at<float>(k + 1, j + 1) * boyImage.at<uchar>(y + k, x + j);
                }
            }
            filteredBoyImage.at<uchar>(y, x) = saturate_cast<uchar>(sum);
        }
    }

    // Display the filtered image
    imshow("Filtered Boy Image", filteredBoyImage);

    // Load the underexposed image in GrayScale mode
    Mat underexposedImage = imread("underexposed.jpg", 0);

    // Check if the underexposed image is loaded properly
    if (underexposedImage.empty()) {
        cout << "Cannot load underexposed image!" << endl;
        return -1;
    }

    // Find the min and max pixel values for the underexposed image
    double minVal, maxVal;
    minMaxLoc(underexposedImage, &minVal, &maxVal);
    cout << "Underexposed MinVal: " << minVal << ", MaxVal: " << maxVal << endl;

    // Stretch the histogram of the underexposed image using custom function
    Mat stretchedImage;
    stretchHistogram(underexposedImage, stretchedImage, minVal, maxVal);

    // Save the stretched image
    imwrite("underexposedStretched.jpg", stretchedImage);

    // Apply histogram equalization to the stretched image using custom function
    Mat equalizedImage;
    equalizeHistogram(stretchedImage, equalizedImage);

    // Save the equalized image
    imwrite("underexposedEqualized.jpg", equalizedImage);

    // Display the original, stretched, and equalized underexposed images
    imshow("Original Underexposed Image", underexposedImage);
    imshow("Stretched Underexposed Image", stretchedImage);
    imshow("Equalized Underexposed Image", equalizedImage);

    waitKey(0);
    return 0;
}

// Implementation des fonctions personnalisées
void calculateHistogram(const Mat& image, Mat& hist, int histSize) {
    hist = Mat::zeros(1, histSize, CV_32F);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int pixelValue = (int)image.at<uchar>(y, x);
            hist.at<float>(pixelValue)++;
        }
    }
}

void stretchHistogram(const Mat& image, Mat& stretchedImage, double minVal, double maxVal) {
    stretchedImage = Mat::zeros(image.size(), image.type());
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            stretchedImage.at<uchar>(y, x) = saturate_cast<uchar>(255.0 * (image.at<uchar>(y, x) - minVal) / (maxVal - minVal));
        }
    }
}

void equalizeHistogram(const Mat& image, Mat& equalizedImage) {
    Mat hist;
    calculateHistogram(image, hist, 256);
    Mat cdf = hist.clone();
    for (int i = 1; i < 256; i++) {
        cdf.at<float>(i) += cdf.at<float>(i - 1);
    }
    cdf *= 255.0 / image.total();
    equalizedImage = Mat::zeros(image.size(), image.type());
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            equalizedImage.at<uchar>(y, x) = saturate_cast<uchar>(cdf.at<float>(image.at<uchar>(y, x)));
        }
    }
}
