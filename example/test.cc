#include "src/parallel.h"
#include "src/safe_stl.h"
#include <string>

// 模拟推理接口
class Infer {
public:
    Infer(std::string& model_path): model_path_(model_path){};
    std::string Predict(int input){
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 模拟耗时
        return "infer" + std::to_string(input);
    }
private:
    std::string model_path_;
}


class MutiInfer : public AutoParallelSimpleInferencePredictor<Infer,std::string,int,std::string>{

public:
    MutiInfer(std::string model_path,int thread_num): AutoParallelSimpleInferencePredictor(model_path,thread_num){};

    std::vector<std::string> Predict(std::vector<int> input){
        std::vector<int> res;
        for(auto& item: input){
            AutoParallelSimpleInferencePredictor::PredictThread(item); // 异步操作
        }
        for(int i = 0; i < input.size(); i++){
            auto result = AutoParallelSimpleInferencePredictor::GetResult();
            res.push_back(result);
        }
    }

private:
    int thread_num_;
}


void Producer(ThreadSafeDeque<int>& dq) {
    int val =0;
    while(true){
        dq.push_back(val);
        std::cout << "[生产者 " << id << "] 推入: " << val << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 模拟耗时
        val++;
    }
}

void Consumer(ThreadSafeDeque<int>& dq, MutiInfer& mutiinfer ){
    std::vector<int> inputs;
    std::vector<int> res;
    while(true){
        int input;
        dq.wait_and_pop_front(input);
        inputs.push_back(inputs);
        if(inputs.size() > 10){
           auto result =  mutiinfer.Predict(inputs);
           res.insert(res.end(),
                   std::make_move_iterator(result.begin()),
                   std::make_move_iterator(result.end()));
           inputs.clear();
        }
    }
}


int main() {
    ThreadSafeDeque<int> dq;

    MutiInfer multi_infer_engine("model_path",10); //开启10个实例同时推理


    std::thread producer_thread(Producer, std::ref(dq));

    std::thread consumer_thread(Consumer, std::ref(dq), std::ref(multi_infer_engine));

    producer_thread.join();
    consumer_thread.join();
    return 0;
}