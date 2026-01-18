#include "utils.hpp"
#include "ilogger.h"

void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }
    
    INFO("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
        "type=%s, qnt_type=%s, "
        "zp=%d, scale=%f\n",
        attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
        attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

std::vector<resnet_input> split_image(cv::Mat& image) {

    std::cout<<image.cols  << image.rows<<std::endl;
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        exit(1);  // 如果加载失败，退出程序
    }

    int img_width = image.cols;
    int img_height = image.rows;

    // 计算每个小块的大小
    int block_width = img_width / COLS;
    int block_height = img_height / ROWS;

    // 存储切割后的图块
    std::vector<resnet_input> inputs;

    // 按照行列顺序切割图像
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            // 计算切割区域的坐标
            int left = j * block_width;
            int top = i * block_height;
            int right = left + block_width;
            int bottom = top + block_height;

            // 确保切割区域不会超出原始图像边界
            right = std::min(right, img_width);
            bottom = std::min(bottom, img_height);

            // 从原图中裁剪出小块
            cv::Rect roi(left, top, right - left, bottom - top);
            cv::Mat img_crop = image(roi);

            // 将小块加入到返回值中
            inputs.push_back(resnet_input(img_crop, i * COLS + j));
        }
    }

    return inputs;
}





std::vector<resnet_input> split_image_main(const std::string& image_path){
    cv::Mat image = cv::imread(image_path);
    std::cout<<image.cols  << image.rows<<std::endl;
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        exit(1);  // 如果加载失败，退出程序
    }

    int img_width = image.cols;
    int img_height = image.rows;

    // 计算每个小块的大小
    int block_width = img_width / 2;
    int block_height = img_height / 2;
    std::vector<resnet_input> inputs_H;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            // 计算切割区域的坐标
            int left = j * block_width;
            int top = i * block_height;
            int right = left + block_width;
            int bottom = top + block_height;

            // 确保切割区域不会超出原始图像边界
            right = std::min(right, img_width);
            bottom = std::min(bottom, img_height);

            // 从原图中裁剪出小块
            cv::Rect roi(left, top, right - left, bottom - top);
            cv::Mat img_crop = image(roi);

            // 将小块加入到返回值中
            inputs_H.push_back(resnet_input(img_crop, i * COLS + j));
        }
    }
    return inputs_H;
}


void split_image_H(resnet_input& input_H,std::queue<resnet_input> HJB,std::mutex& mutex_H){
    cv::Mat image = input_H.img;
    
    std::cout<<image.cols  << image.rows<<std::endl;
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        exit(1);  // 如果加载失败，退出程序
    }

    int img_width = image.cols;
    int img_height = image.rows;

    // 计算每个小块的大小
    int block_width = img_width /ROWS;
    int block_height = img_height / COLS;
    std::vector<resnet_input> inputs_H;
    for (int i = 0; i < ROWS/2; ++i) {
        for (int j = 0; j < COLS/2; ++j) {
            // 计算切割区域的坐标
            int left = j * block_width;
            int top = i * block_height;
            int right = left + block_width;
            int bottom = top + block_height;

            // 确保切割区域不会超出原始图像边界
            right = std::min(right, img_width);
            bottom = std::min(bottom, img_height);

            // 从原图中裁剪出小块
            cv::Rect roi(left, top, right - left, bottom - top);
            cv::Mat img_crop = image(roi);

            // 将小块加入到返回值中
            {
                std::unique_lock<std::mutex> U_H(mutex_H);
                HJB.push(resnet_input(img_crop, i * COLS + j));
            }
        }
    }
} 


resnet_input get_resnet_input(std::queue<resnet_input> HJB,std::mutex& mutex_H){
    std::unique_lock<std::mutex> U_H(mutex_H);
    while(HJB.empty()){
        
    }
    if(!HJB.empty()){
        resnet_input input = HJB.front();
        HJB.pop();
    return input;
}
}