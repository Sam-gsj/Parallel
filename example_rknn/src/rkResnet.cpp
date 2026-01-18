#include "rkResnet.hpp"
#include <stdio.h>
#include <mutex>
#include "rknn_api.h"


#include "preprocess.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "coreNum.hpp"
#include "rkResnet.hpp"
#include "utils.hpp"
#include "ilogger.h"

int  print_once = 0;
rkResnet::rkResnet(const std::string &model_path)
{
    this->model_path = model_path;
}

int rkResnet::init(rknn_context *ctx_in, bool share_weight)
{

    if(!share_weight){
        INFO("Loading model...\n");
    }
    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);
    // 模型参数复用/Model parameter reuse
    if (share_weight == true){
        ret = rknn_dup_context(ctx_in, &ctx);
    }
    else{
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    }
    if (ret < 0)
    {
        INFOE("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 设置模型绑定的核心/Set the core of the model that needs to be bound
    rknn_core_mask core_mask;
    switch (get_core_num())
    {
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    }
    ret = rknn_set_core_mask(ctx, core_mask);  //多核
    if (ret < 0)
    {
        INFOE("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        INFOE("rknn_init error ret=%d\n", ret);
        return -1;
    }
    

    // 获取模型输入输出参数/Obtain the input and output parameters of the model
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        INFOE("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 设置输入参数/Set the input parameters
    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            INFOE("rknn_init error ret=%d\n", ret);
            return -1;
        }
    }

    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            INFOE("rknn_init error ret=%d\n", ret);
            return -1;
        }
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW )
    {
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else if(input_attrs[0].fmt == RKNN_TENSOR_NHWC )
    {
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    
    if (!share_weight)  // 仅当第一次调用时打印
    {
        INFO("model input height=%d, width=%d, channel=%d\n", height, width, channel);
        INFO("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
        dump_tensor_attr(&(input_attrs[0]));
        dump_tensor_attr(&(output_attrs[0]));
        INFO("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
    }

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    return 0;
}




rknn_context *rkResnet::get_pctx()
{
    return &ctx;
}


resnet_results rkResnet::Predict(resnet_input& input)
{
    std::lock_guard<std::mutex> lock(mtx);
    cv::Mat img;
    cv::cvtColor(input.img, img, cv::COLOR_BGR2RGB);
    img_width = img.cols;
    img_height = img.rows;

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    cv::Size target_size(width, height);  //这里设置是模型的输入和输出
    cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
    // 计算缩放比例/Calculate the scaling ratio
    float scale_w = (float)target_size.width / img.cols;
    float scale_h = (float)target_size.height / img.rows;

    // 图像缩放/Image scaling
    if (img_width != width || img_height != height)
    {
        //******************rga有bug************************** */
        
        // rga
        // rga_buffer_t src;
        // rga_buffer_t dst;
        // memset(&src, 0, sizeof(src));
        // memset(&dst, 0, sizeof(dst));
        // ret = resize_rga(src, dst, img, resized_img, target_size);
        // if (ret != 0)
        // {
        //     fprintf(stderr, "resize with rga error\n");
        // }

        // opencv
        float min_scale = std::min(scale_w, scale_h);
        scale_w = min_scale;
        scale_h = min_scale;
        letterbox(img, resized_img, pads, min_scale, target_size);

        inputs[0].buf = resized_img.data;
    }
    else
    {
        inputs[0].buf = img.data;
    }

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 1;  //反量化
    }

    // 模型推理/Model inference
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    // 后处理/Post-processing


    softmax((float*)outputs[0].buf, output_attrs[0].n_elems);
    resnet_results results;
    results.id = input.id;
    get_topk_with_indices((float*)outputs[0].buf, output_attrs[0].n_elems, results);

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

    return results;
}

rkResnet::~rkResnet()
{
    deinitPostProcess();

    ret = rknn_destroy(ctx);

    if (model_data)
        free(model_data);

    if (input_attrs)
        free(input_attrs);
    if (output_attrs)
        free(output_attrs);
}
