#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <queue>
#include <thread>
#include <mutex>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkResnet.hpp"
#include "const.hpp"
#include "utils.hpp"
#include "postprocess.h"
#include "ilogger.h"
#include "src/parallel.h" 

// 假设 rkResnet 已经被修改为符合框架要求：
// 1. 构造函数 rkResnet(const std::string& model_path)
// 2. 方法 resnet_results Predict(const resnet_input& input)
using AutoRKNN = AutoParallelSimpleInferencePredictor<rkResnet, const std::string &, resnet_input, resnet_results>;

int main(int argc, char** argv) {

    std::string model_path = "/home/orangepi/parallel/example_rknn/model/lenet5_32.rknn";
    std::string image_path = "/home/orangepi/parallel/example_rknn/picture/pingdi.jpg";
    int thread_num = 1; 

    // --- 读取图片 ---
    cv::Mat input_image = cv::imread(image_path);
    if (input_image.empty()) {
        std::cerr << "Error: Load image failed: " << image_path << std::endl;
        return -1;
    }
    std::cout << "Image loaded: " << input_image.cols << "x" << input_image.rows << std::endl;

    // --- 计时开始 ---
    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    try {
        // 1. 图像预处理/切片
        std::vector<resnet_input> inputs = split_image(input_image);
        size_t task_count = inputs.size();
        std::cout << "Split into " << task_count << " tasks." << std::endl;

        // 【关键修复】检查并修复空图像
        // 这解决了 "Assertion failed ... resize" 的崩溃问题
        for (size_t i = 0; i < task_count; ++i) {
            if (inputs[i].img.empty() || inputs[i].img.cols == 0 || inputs[i].img.rows == 0) {
                // 如果是空图，用 32x32 的黑图替换 (尺寸根据模型输入调整)
                std::cout << "******" <<std::endl;
                inputs[i].img = cv::Mat::zeros(32, 32, CV_8UC3); 
            }
        }

        // 2. 初始化并行推理器
        // 这里直接使用 rkResnet，假设 rkResnet 类已经支持 rkResnet(string) 构造函数
        AutoRKNN predictor(model_path, thread_num);

        // 3. 提交任务
        for (const auto& input : inputs) {
            // 调用框架的提交接口
            predictor.PredictThread(input);
        }

        // 4. 获取结果
        std::vector<resnet_results> results_vec;
        results_vec.reserve(task_count);

        for (size_t i = 0; i < task_count; ++i) {
            resnet_results res;
            if (predictor.GetResult(res)) {
                results_vec.push_back(res);
            } else {
                // std::cerr << "Error: Failed to get result for task " << i << std::endl;
                results_vec.push_back(resnet_results()); // 插入空结果占位
            }
        }

        // 5. 后处理/合成
        auto outputs = synthesize_image(inputs, results_vec);

        // --- 计时结束 ---
        gettimeofday(&time, nullptr);
        auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        float cost = float(endTime - startTime);

        printf("--------------------------------\n");
        printf("Processed %zu blocks.\n", task_count);
        printf("Total Time: %.2f ms\n", cost);
        printf("FPS       : %.2f\n", 1000.0 / cost);
        printf("--------------------------------\n");

        // 6. 保存结果
        if (!outputs.empty()) {
            std::string save_path = "output_0.jpg";
            if (outputs.size() > 3) {
                cv::imwrite(save_path, outputs[3]);
                std::cout << "Saved " << save_path << std::endl;
            } else {
                cv::imwrite(save_path, outputs[0]);
                std::cout << "Saved first output image." << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}