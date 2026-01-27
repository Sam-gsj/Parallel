#pragma once

#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "const.hpp"
#include <mutex>
#include <string>

class rkResnet
{
private:
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data = nullptr;
    rknn_context ctx = 0;
    
    rknn_input_output_num io_num = {0};
    rknn_tensor_attr *input_attrs = nullptr;
    rknn_tensor_attr *output_attrs = nullptr;
    rknn_input inputs[1];

    int channel = 0, width = 0, height = 0;
    int img_width = 0, img_height = 0;

public:
    // 构造函数
    rkResnet(const std::string &model_path);
    rkResnet(const rkResnet&) = delete;
    rkResnet& operator=(const rkResnet&) = delete;

    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    resnet_results Predict(resnet_input& input);
    ~rkResnet();
};