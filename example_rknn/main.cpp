#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <unistd.h>

#include "rkResnet.hpp"
#include "const.hpp"
#include "utils.hpp"
#include "postprocess.h"
#include "src/parallel.h" 


using AutoRKNN = AutoParallelSimpleInferencePredictor<rkResnet, const std::string &, resnet_input, resnet_results>;

int main(int argc, char** argv) {

    std::string model_path = "/home/orangepi/parallel/example_rknn/model/lenet5_32.rknn";
    std::string image_path = "/home/orangepi/parallel/example_rknn/picture/pingdi.jpg";
    int thread_num = 3; // NPU 通常 3 核并行效率最高

    cv::Mat input_image = cv::imread(image_path);
    if (input_image.empty()) {
        std::cerr << "Error: Load image failed." << std::endl;
        return -1;
    }

    // 计时
    struct timeval time;
    gettimeofday(&time, nullptr);
    

    try {
        std::vector<resnet_input> inputs = split_image(input_image);
        size_t task_count = inputs.size();
        std::cout << "Tasks: " << task_count << std::endl;


        for (auto& item : inputs) {
            if (item.img.empty() || item.img.cols == 0) {
                item.img = cv::Mat::zeros(32, 32, CV_8UC3);
            }
        }

        std::cout << "Initializing AutoRKNN with " << thread_num << " threads..." << std::endl;
        AutoRKNN predictor(model_path, thread_num);
        
        auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        std::cout << "Submitting tasks..." << std::endl;
        for (const auto& input : inputs) {
            predictor.PredictThread(input);
        }

        std::vector<resnet_results> results_vec;
        results_vec.reserve(task_count);

        std::cout << "Waiting for results..." << std::endl;
        for (size_t i = 0; i < task_count; ++i) {
            resnet_results res;
        
            if (predictor.GetResult(res)) {
                results_vec.push_back(res);
            } else {
                resnet_results dummy;
                dummy.id = inputs[i].id; 
                results_vec.push_back(dummy);
            }
        }

        // 6. 后处理与保存
        auto outputs = synthesize_image(inputs, results_vec);

        gettimeofday(&time, nullptr);
        auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;
        float cost = float(endTime - startTime);

        printf("--------------------------------\n");
        printf("Mode      : AutoParallel\n");
        printf("Processed : %zu blocks\n", task_count);
        printf("Total Time: %.2f ms\n", cost);
        printf("FPS       : %.2f\n", 1000.0 / cost);
        printf("--------------------------------\n");

        if (!outputs.empty()) {
            cv::imwrite("output_auto.jpg", outputs.size() > 3 ? outputs[3] : outputs[0]);
        }

    } catch (const std::exception& e) {
        std::cerr << "CRASH Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}