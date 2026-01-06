#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include "parallel.h"

// --- 模拟耗时 100ms 的推理 ---
const int DELAY_MS = 100;

struct MockParams {};
using MockInput = int;
using MockResult = int;

class MockPredictor {
public:
    MockPredictor(const MockParams&) {}
    MockResult Predict(const MockInput& in) {
        std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_MS));
        return in * 2;
    }
};

// --- 串行基准测试 ---
void RunSerial(int tasks) {
    MockParams params;
    MockPredictor predictor(params);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < tasks; ++i) {
        predictor.Predict(i);
    }
    auto end = std::chrono::steady_clock::now();
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[Serial  ] Tasks: " << tasks << " | Time: " << ms << " ms | QPS: " 
              << (tasks * 1000.0 / ms) << std::endl;
}


void RunParallel(int tasks, int threads) {
    MockParams params;
    using AutoPredictor = AutoParallelSimpleInferencePredictor<MockPredictor, MockParams, MockInput, MockResult>;
    AutoPredictor parallel_predictor(params, threads);

    auto start = std::chrono::steady_clock::now();
    

    for (int i = 0; i < tasks; ++i) {
        parallel_predictor.PredictThread(i);
    }
    
    MockResult res;
    for (int i = 0; i < tasks; ++i) {
        parallel_predictor.GetResult(res);
    }
    
    auto end = std::chrono::steady_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[Parallel] Tasks: " << tasks << " | Time: " << ms << " ms | QPS: " 
              << (tasks * 1000.0 / ms) << " (Threads: " << threads << ")" << std::endl;
}

int main() {
    int tasks = 200;
    int threads = 8;

    std::cout << "=== Performance Comparison (Task Cost: " << DELAY_MS << "ms) ===" << std::endl;
    
    RunSerial(tasks);
    RunParallel(tasks, threads);
    
    std::cout << "==========================================================" << std::endl;
    return 0;
}