// Author: sumingrui

#ifndef UTILS_H_
#define UTILS_H_

#include <opencv2/opencv.hpp>
#include <string>

cv::Mat norm(cv::Mat mat);
cv::Mat bf_pad(cv::Mat mat, int nChannels, int kernel_size);
void pad_2dcnn(cv::Mat &mat);
void save_xml(cv::Mat mat, std::string path);
void save_imagesc(cv::Mat img, std::string path);

#endif // UTILS_H_