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

#include <iostream>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>

template <typename T>
class ThreadSafeDeque {
private:
    std::deque<T> deque_;             
    mutable std::mutex mtx_;            
    std::condition_variable data_cond_; 

public:
    ThreadSafeDeque() = default;
    
    ThreadSafeDeque(const ThreadSafeDeque&) = delete;
    ThreadSafeDeque& operator=(const ThreadSafeDeque&) = delete;

    // push_back
    void push_back(const T& value) {
        std::lock_guard<std::mutex> lock(mtx_);
        deque_.push_back(value);
        data_cond_.notify_one(); 
    }

    void push_front(const T& value) {
        std::lock_guard<std::mutex> lock(mtx_);
        deque_.push_front(value);
        data_cond_.notify_one();
    }

    void wait_and_pop_front(T& value) {
        std::unique_lock<std::mutex> lock(mtx_);

        data_cond_.wait(lock, [this]{ return !deque_.empty(); });
        
        value = std::move(deque_.front()); 
        deque_.pop_front();
    }

    // pop back
    void wait_and_pop_back(T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        data_cond_.wait(lock, [this]{ return !deque_.empty(); });
        
        value = std::move(deque_.back());
        deque_.pop_back();
    }
    // pop front
    bool try_pop_front(T& value) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (deque_.empty()) {
            return false;
        }
        value = std::move(deque_.front());
        deque_.pop_front();
        return true;
    }

    // else 

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return deque_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return deque_.size();
    }
};

