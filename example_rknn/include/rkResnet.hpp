#pragma once

#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "const.hpp"






class rkResnet
{
private:
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];

    int channel, width, height;
    int img_width, img_height;

    float nms_threshold, box_conf_threshold;

public:
    rkResnet(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild) ;
    rknn_context *get_pctx();
    resnet_results Predict(resnet_input& input);
    ~rkResnet();
};
