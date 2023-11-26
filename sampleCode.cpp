#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

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

    // Calculate and normalize the histogram for the boy image
    int histSize = 256; // number of bins
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat boyHist;
    calcHist(&boyImage, 1, 0, Mat(), boyHist, 1, &histSize, &histRange, true, false);
    normalize(boyHist, boyHist, 0, 400, NORM_MINMAX, -1, Mat());

    // Draw the histogram for the boy image
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat boyHistImage(hist_h, hist_w, CV_8UC1, Scalar(0));
    for (int i = 0; i < histSize; i++) {
        line(boyHistImage,
            Point(bin_w * i, hist_h),
            Point(bin_w * i, hist_h - cvRound(boyHist.at<float>(i))),
            Scalar(255),
            bin_w);
    }

    // Load the underexposed image in GrayScale mode
    Mat underexposedImage = imread("underexposed.jpg", 0);

    // Check if the underexposed image is loaded properly
    if (underexposedImage.empty()) {
        cout << "Cannot load underexposed image!" << endl;
        return -1;
    }



    // Find the min and max pixel values and their locations for the underexposed image
    double minVal, maxVal;
    minMaxLoc(underexposedImage, &minVal, &maxVal);
    cout << "Underexposed MinVal: " << minVal << ", MaxVal: " << maxVal << endl;

    // Stretch the histogram of the underexposed image
    Mat stretchedImage;
    underexposedImage.convertTo(stretchedImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    // Save the stretched image
    imwrite("underexposedStretched.jpg", stretchedImage);

    // Apply histogram equalization to the stretched image
    Mat equalizedImage;
    equalizeHist(stretchedImage, equalizedImage);

    // Save the equalized image
    imwrite("underexposedEqualized.jpg", equalizedImage);

    // Display the boy image and its histogram
    imshow("Boy Image", boyImage);
    imshow("Histogram of Boy Image", boyHistImage);

    // Display the original, stretched, and equalized underexposed images
    imshow("Original Underexposed Image", underexposedImage);
    imshow("Stretched Underexposed Image", stretchedImage);
    imshow("Equalized Underexposed Image", equalizedImage);

    waitKey(0);
    return 0;
}
