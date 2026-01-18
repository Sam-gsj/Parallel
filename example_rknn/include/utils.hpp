#pragma once


#include <vector>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "const.hpp"
#include <queue>
#include <mutex>

void dump_tensor_attr(rknn_tensor_attr *attr);
unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
unsigned char *load_model(const char *filename, int *model_size);
int saveFloat(const char *file_name, float *output, int element_size);
std::vector<resnet_input> split_image(cv::Mat& image);


std::vector<resnet_input> split_image_main(const std::string& image_path);
void split_image_H(resnet_input& input_H,std::queue<resnet_input> HJB,std::mutex& mutex_H);
resnet_input get_resnet_input(std::queue<resnet_input> HJB,std::mutex& mutex_H);