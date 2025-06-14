/**
 * @file      calibrator.cpp
 * @brief     
 *
 * Copyright (c) 2024 
 *
 * @author    shiningchen
 * @date      2025.06.13
 * @version   1.0
*/


#include <iostream>
#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>

#include "calibrator.h"
#include "utils.h"

using namespace nvinfer1;


/**
 * @brief 预处理图像数据，将其调整为指定大小，并进行归一化处理。
 *
 * 该函数首先对输入图像进行letterbox缩放，然后将其调整为指定的大小，
 * 并将图像数据从HWC格式转换为CHW格式，从BGR格式转换为RGB格式，
 * 最后对像素值进行归一化处理。
 *
 * @param img 输入图像，以OpenCV的Mat格式表示。
 * @param input_w 目标图像的宽度。
 * @param input_h 目标图像的高度。
 * @return 返回一个包含预处理后图像数据的浮点型向量。
 */
std::vector<float> preprocess(cv::Mat& img, int input_w, int input_h)
{
    int elements {3 * input_h * input_w};

    // letterbox
    int w, h, x, y;
    float r_w {input_w / (img.cols * 1.0)};
    float r_h {input_h / (img.rows * 1.0)};
    // 如果原图的宽高比大于输入图像的宽高比，则将原图缩放到输入图像的宽度，并计算缩放后的高度
    if (r_h > r_w){
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    // 如果原图的宽高比小于输入图像的宽高比，则将原图缩放到输入图像的高度，并计算缩放后的宽度
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    // 创建一个与输入图像大小相同的空白图像
    cv::Mat re(h, w, CV_8UC3);
    // 将原图缩放到与输入图像大小相同的图像
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    // 创建一个与输入图像大小相同的空白图像，并用灰度值128填充
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    // 将缩放后的图像复制到空白图像的指定位置
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    // HWC to CHW , BGR to RGB, Normalize
    std::vector<float> result(elements);
    float* norm_data = result.data();  // normalized data
    uchar* uc_pixel = out.data;
    // 遍历输入图像的每个像素，进行归一化处理
    for (int i {0}; i < input_h * input_w; i++)
    {
        norm_data[i] = (float)uc_pixel[2] / 255.0;
        norm_data[i + input_h * input_w] = (float)uc_pixel[1] / 255.0;
        norm_data[i + 2 * input_h * input_w] = (float)uc_pixel[0] / 255.0;
        uc_pixel += 3;
    }

    return result;
}


Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batch_size, int input_w, int input_h, const char* img_dir, const char* calib_table_name, bool read_cache)
    : batch_size_(batch_size)
    , input_w_(input_w)
    , input_h_(input_h)
    , img_idx_(0)
    , img_dir_(img_dir)
    , calib_table_name_(calib_table_name)
    , read_cache_(read_cache)
{
    // 计算输入数据的数量
    input_count_ = 3 * input_w * input_h * batch_size;
    batch_data = new float[input_count_];
    cudaMalloc(&device_input_, input_count_ * sizeof(float));
    readFilesInDir(img_dir, img_files_);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    cudaFree(device_input_);
    if (batch_data) {
        delete[] batch_data;
        batch_data = nullptr;
    }
    device_input_ = nullptr;

}

int Int8EntropyCalibrator2::getBatchSize() const noexcept
{
    return batch_size_;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    // 如果img_idx_加上batch_size_大于img_files_的大小，则返回false
    if (img_idx_ + batch_size_ > (int)img_files_.size()) { return false; }

    // 指向batch_data的指针
    float *ptr = batch_data;
    // 遍历img_idx_到img_idx_ + batch_size_之间的图片
    for (int i = img_idx_; i < img_idx_ + batch_size_; i++)
    {
        // 输出图片路径和索引
        std::cout << img_files_[i] << "  " << i << std::endl;
        // 读取图片
        cv::Mat temp = cv::imread(img_dir_ + "/" + img_files_[i], cv::IMREAD_COLOR);
        // 如果图片为空，则输出错误信息并返回false
        if (temp.empty()){
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        // 预处理图片
        std::vector<float> input_data = preprocess(temp, input_w_, input_h_);
        // 将预处理后的数据复制到batch_data中
        memcpy(ptr, input_data.data(), (int)(input_data.size()) * sizeof(float));
        // 指针向后移动
        ptr += input_data.size();
    }
    // 更新img_idx_
    img_idx_ += batch_size_;

    // 将batch_data中的数据复制到device_input_中
    cudaMemcpy(device_input_, batch_data, input_count_ * sizeof(float), cudaMemcpyHostToDevice);
    // 将device_input_绑定到bindings[0]上
    bindings[0] = device_input_;
    
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) noexcept
{
    // 打印读取校准缓存的信息
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    // 清空校准缓存
    calib_cache_.clear();
    // 打开校准缓存文件
    std::ifstream input(calib_table_name_, std::ios::binary);
    // 不跳过空白字符
    input >> std::noskipws;
    // 如果校准缓存存在且文件状态良好
    if (read_cache_ && input.good())
    {
        // 将文件内容复制到校准缓存中
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    // 设置校准缓存长度
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    // 输出写入校准缓存的信息
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    // 创建一个二进制输出流
    std::ofstream output(calib_table_name_, std::ios::binary);
    // 将缓存写入输出流
    output.write(reinterpret_cast<const char*>(cache), length);
}
