// Author: sumingrui

#ifndef TF_H_
#define TF_H_

#include <string>
#include <opencv2/opencv.hpp>

using std::string;

bool ReadRawfile(string rawfilepath, cv::Mat &img_cube);
// bool Draw_gt(string matfilepath, string key);
int TF_2dcnn(const char* rawfilepath,const char* filename, const char* sendImgPath);
// int TF_3dcnn(string rawfilepath,string filename, string & sendImgPath);

#endif // TF_H_
