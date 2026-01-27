#include "rkResnet.hpp"
#include <stdio.h>
#include <mutex>
#include "rknn_api.h"
#include "preprocess.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "coreNum.hpp"
#include "utils.hpp"
#include "ilogger.h" // 假设你有这个日志库

// 构造函数
rkResnet::rkResnet(const std::string &model_path)
{
    this->model_path = model_path;
    
    // 【修改3】构造时立即初始化，适应 main 函数的逻辑
    // 如果是单线程串行，传入 nullptr 和 false
    int ret = this->init(nullptr, false);
    if (ret != 0) {
        printf("Error: Model initialization failed in constructor!\n");
        // 这里可以抛出异常，或者设置一个标志位
    }
}

int rkResnet::init(rknn_context *ctx_in, bool share_weight)
{
    if(!share_weight){
        printf("Loading model...\n");
    }
    
    int model_data_size = 0;
    // 确保 load_model 实现正确，返回分配的内存指针
    model_data = load_model(model_path.c_str(), &model_data_size);
    if (model_data == nullptr) {
        printf("Error: load_model failed.\n");
        return -1;
    }

    // 模型参数复用/Model parameter reuse
    if (share_weight == true && ctx_in != nullptr){
        ret = rknn_dup_context(ctx_in, &ctx);
    }
    else{
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    }
    
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 设置核心 (保留你的逻辑)
    rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO; // 建议默认 AUTO，或者保留你的 switch
    switch (get_core_num()) {
        case 0: core_mask = RKNN_NPU_CORE_0; break;
        case 1: core_mask = RKNN_NPU_CORE_1; break;
        case 2: core_mask = RKNN_NPU_CORE_2; break;
    }
    rknn_set_core_mask(ctx, core_mask);

    // 获取SDK版本 (可选，略)

    // 获取 IO 数量
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) return -1;

    // 分配并获取 Input Attr
    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    }

    // 分配并获取 Output Attr
    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    // 解析尺寸
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    } else { // NHWC
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    if (!share_weight) {
        printf("Model initialized: %dx%dx%d\n", width, height, channel);
    }

    // 初始化 inputs 结构体
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    return 0;
}

rknn_context *rkResnet::get_pctx() { return &ctx; }

resnet_results rkResnet::Predict(resnet_input& input)
{
    // 检查 ctx 是否有效
    if (ctx == 0) {
        printf("Error: Context is null in Predict.\n");
        return resnet_results();
    }

    std::lock_guard<std::mutex> lock(mtx);
    
    cv::Mat img;
    // 确保输入不为空
    if (input.img.empty()) {
         return resnet_results();
    }

    // 转换颜色
    cv::cvtColor(input.img, img, cv::COLOR_BGR2RGB);

    // 【修改4】处理尺寸不匹配问题
    // 你的代码之前在 img_width != width 时才处理，这很好。
    // 但必须确保 inputs[0].buf 指向的数据是有效的且大小正确。
    
    cv::Mat resized_img;
    
    if (img.cols != width || img.rows != height) {
        // 简单处理：直接 resize (Letterbox 更好，但为了排查 crash，先用最稳的)
        // 注意：resized_img 是局部变量，它的 data 指针只在 Predict 函数内有效
        // 只要 rknn_run 在函数返回前执行完毕即可。
        cv::resize(img, resized_img, cv::Size(width, height));
        inputs[0].buf = resized_img.data;
    } else {
        inputs[0].buf = img.data;
    }

    // 设置输入
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    // 准备输出 buffer
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 1;
    }

    // 推理
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        printf("rknn_run failed %d\n", ret);
        return resnet_results();
    }
    
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    // 后处理
    // 假设 softmax 和 get_topk_with_indices 是安全的
    if(output_attrs && outputs[0].buf) {
        softmax((float*)outputs[0].buf, output_attrs[0].n_elems);
    }
    
    resnet_results results;
    results.id = input.id;
    
    if(outputs[0].buf) {
        get_topk_with_indices((float*)outputs[0].buf, output_attrs[0].n_elems, results);
    }

    rknn_outputs_release(ctx, io_num.n_output, outputs);

    return results;
}

rkResnet::~rkResnet()
{
    if (ctx > 0) {
        rknn_destroy(ctx);
        ctx = 0;
    }

    if (model_data != nullptr) {
        free(model_data);
        model_data = nullptr;
    }

    if (input_attrs != nullptr) {
        free(input_attrs);
        input_attrs = nullptr;
    }
    
    if (output_attrs != nullptr) {
        free(output_attrs);
        output_attrs = nullptr;
    }
}