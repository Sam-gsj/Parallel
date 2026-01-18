// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2025 guoshengjian Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>
#include <future>
#include <thread>

#include "thread_pool.h"

template <typename Predictor, typename PredictorParams, typename PredictorInput,
          typename PredictorResult>
class AutoParallelSimpleInferencePredictor {
private:
  struct InferenceInstance {
    std::shared_ptr<Predictor> Predictor_;
    std::queue<PredictorInput> task_queue;
    std::queue<std::promise<PredictorResult>> promise_queue;
    std::mutex queue_mutex;
    std::atomic<bool> is_busy{false};
    int instance_id;
  };

public:
  AutoParallelSimpleInferencePredictor(const PredictorParams &params, int thread_num);
  
  bool Init();

  std::future<PredictorResult> PredictAsync(const PredictorInput &input);


  bool PredictThread(const PredictorInput &input);
  

  bool GetResult(PredictorResult& result_out);

  virtual ~AutoParallelSimpleInferencePredictor();

private:
  void ProcessInstanceTasks(int instance_id);
  PredictorParams params_;
  int thread_num_;

  std::atomic<int> round_robin_index_{0};
  std::unique_ptr<PaddlePool::ThreadPool> pool_;
  std::vector<std::unique_ptr<InferenceInstance>> instances_;

  std::queue<std::future<PredictorResult>> legacy_results_;
  std::mutex legacy_results_mutex_;
};


template <typename Predictor, typename PredictorParams, typename PredictorInput,
          typename PredictorResult>
AutoParallelSimpleInferencePredictor<Predictor, PredictorParams, PredictorInput,
                                    PredictorResult>::
    AutoParallelSimpleInferencePredictor(const PredictorParams &params, int thread_num)
    : params_(params), thread_num_(thread_num) {
  if (thread_num_ > 1) {
    if (!Init()) {
      std::cerr << "Predictor pool init error." << std::endl;
      exit(-1);
    }
  }
}

template <typename Predictor, typename PredictorParams, typename PredictorInput,
          typename PredictorResult>
bool AutoParallelSimpleInferencePredictor<Predictor, PredictorParams, PredictorInput,
                                    PredictorResult>::Init() {
  try {
    pool_ = std::unique_ptr<PaddlePool::ThreadPool>(
        new PaddlePool::ThreadPool(thread_num_));

    for (int i = 0; i < thread_num_; i++) {
      auto instance =
          std::unique_ptr<InferenceInstance>(new InferenceInstance());
      instance->instance_id = i;

      instance->Predictor_ = std::shared_ptr<Predictor>(new Predictor(params_));

      instances_.push_back(std::move(instance));
    }
  } catch (const std::exception &e) {
    std::cerr << "Init failed: " << e.what() << std::endl;
    return false;
  }
  return true;
}

template <typename Predictor, typename PredictorParams, typename PredictorInput,
          typename PredictorResult>
std::future<PredictorResult> AutoParallelSimpleInferencePredictor<
    Predictor, PredictorParams, PredictorInput,
    PredictorResult>::PredictAsync(const PredictorInput &input) {
  int instance_id = round_robin_index_.fetch_add(1) % thread_num_;
  auto &instance = instances_[instance_id];

  std::promise<PredictorResult> promise;
  auto future = promise.get_future();

  {
    std::lock_guard<std::mutex> lock(instance->queue_mutex);
    instance->task_queue.push(input);
    instance->promise_queue.push(std::move(promise));
  }

  bool expected = false;
  if (instance->is_busy.compare_exchange_strong(expected, true)) {
    pool_->submit([this, instance_id]() { ProcessInstanceTasks(instance_id); });
  }

  return future;
}

template <typename Predictor, typename PredictorParams, typename PredictorInput,
          typename PredictorResult>
void AutoParallelSimpleInferencePredictor<
    Predictor, PredictorParams, PredictorInput,
    PredictorResult>::ProcessInstanceTasks(int instance_id) {
  auto &instance = instances_[instance_id];

  while (true) {
    std::promise<PredictorResult> promise;
    
    std::unique_ptr<PredictorInput> input_ptr;

    {
      std::lock_guard<std::mutex> lock(instance->queue_mutex);
      if (instance->task_queue.empty()) {
        instance->is_busy = false;

        if (!instance->task_queue.empty()) {
          bool expected = false;
          if (instance->is_busy.compare_exchange_strong(expected, true)) {
            continue;
          }
        }
        return;
      }
      
      
      input_ptr.reset(new PredictorInput(std::move(instance->task_queue.front())));
      instance->task_queue.pop();
      
      promise = std::move(instance->promise_queue.front());
      instance->promise_queue.pop();
    } 

    try {
    
      PredictorResult result = instance->Predictor_->Predict(*input_ptr);
      promise.set_value(std::move(result));
    } catch (const std::exception &e) {
      promise.set_exception(std::current_exception());
    }
  }
}

template <typename Predictor, typename PredictorParams, typename PredictorInput,
          typename PredictorResult>
bool AutoParallelSimpleInferencePredictor<
    Predictor, PredictorParams, PredictorInput,
    PredictorResult>::PredictThread(const PredictorInput &input) {
  try {
    auto future = PredictAsync(input);

    std::lock_guard<std::mutex> lock(legacy_results_mutex_);
    legacy_results_.push(std::move(future));

    return true;
  } catch (const std::exception &e) {
    std::cerr << "Failed to submit inference: " << e.what() << std::endl;
    return false;
  }
}

template <typename Predictor, typename PredictorParams, typename PredictorInput,
          typename PredictorResult>
bool AutoParallelSimpleInferencePredictor<Predictor, PredictorParams, PredictorInput,
                                    PredictorResult>::GetResult(PredictorResult& result_out) {
  std::lock_guard<std::mutex> lock(legacy_results_mutex_);

  if (legacy_results_.empty()) {
    return false; 
  }

  try {
    auto future = std::move(legacy_results_.front());
    legacy_results_.pop();

    result_out = future.get(); 
    return true;
  } catch (const std::exception &e) {
    // std::cerr << "Failed to get inference result: " << e.what() << std::endl;
    return false;
  }
}

template <typename Predictor, typename PredictorParams, typename PredictorInput,
          typename PredictorResult>
AutoParallelSimpleInferencePredictor<
    Predictor, PredictorParams, PredictorInput,
    PredictorResult>::~AutoParallelSimpleInferencePredictor() {
  
  while (!legacy_results_.empty()) {
    try {
      legacy_results_.front().get();
    } catch (...) {
    }
    legacy_results_.pop();
  }

  for (auto &instance : instances_) {
    while (instance->is_busy.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}