/**
 * @file      infer.cpp
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

#include <NvOnnxParser.h>

#include "infer.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calibrator.h"
#include "utils.h"

using namespace nvinfer1;


YoloDetector::YoloDetector(
        const std::string trt_file,
        int gpu_id,
        float nms_thresh,
        float conf_thresh,
        int num_class
    ): trt_file_(trt_file), nms_thresh_(nms_thresh), conf_thresh_(conf_thresh), num_class_(num_class),input_element_size(4U),output_element_size(4U)
{
    // 设置日志级别
    g_logger = Logger(ILogger::Severity::kERROR);
    // 设置GPU设备
    cudaSetDevice(gpu_id);

    // 创建CUDA流
    CHECK(cudaStreamCreate(&stream));

    // 加载引擎
    getEngine();

    // 创建执行上下文
    context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    // 获取输入输出张量信息
    for(int32_t i{0}; i < engine->getNbIOTensors(); ++i){
        auto tensor_name = engine->getIOTensorName(i);
        Dims dim = engine->getTensorShape(tensor_name);
        if(engine->getTensorIOMode(tensor_name) == TensorIOMode::kINPUT){
            input_tensors[tensor_name] = dim;
        }else{
            output_tensors[tensor_name] = dim;
        }
    }
    
    
    input_element_size = dataTypeToSize(engine->getTensorDataType(input_tensors.begin()->first));
    output_element_size = dataTypeToSize(engine->getTensorDataType(output_tensors.begin()->first));
    
    Dims input_dims = input_tensors.begin()->second;
    Dims out_dims = output_tensors.begin()->second;

    

    // 计算输入大小
    size_t input_size{input_element_size};
    for(int32_t i{0}; i < input_dims.nbDims; ++i){
        input_size *= input_dims.d[i];
    }

    // 计算输出大小
    size_t output_size{output_element_size};  // 84 * 8400
    for (int32_t i {0}; i < out_dims.nbDims; ++i){
        output_size *= out_dims.d[i];
    }
    output_candidates = out_dims.d[2];

    // 创建输出缓冲区
    output_data = new float[1 + KMaxNumOutputBbox * KNumBoxElement];

    v_buffer_d.resize(2, nullptr);
    CHECK(cudaMalloc(&v_buffer_d[0], input_size));
    CHECK(cudaMalloc(&v_buffer_d[1], output_size));

    // 创建transpose_device和decode_device
    CHECK(cudaMalloc(&transpose_device, output_size ));
    CHECK(cudaMalloc(&decode_device, (1 + KMaxNumOutputBbox * KNumBoxElement) * sizeof(float)));
}

void YoloDetector::getEngine(){
    // 检查引擎文件是否存在
    if (access(trt_file_.c_str(), F_OK) == 0){

        // 读取引擎文件
        std::vector<char> engine_data;
        size_t fsize = readEngine(trt_file_, engine_data);

        // 创建推理运行时
        runtime =  std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), fsize));
        if (engine == nullptr) { std::cout << "Failed loading engine!" << std::endl; return; }
        std::cout << "Succeeded loading engine!" << std::endl;

    } else {
        std::cout << "Engine file not found! Use onnx to create engine!" << std::endl;
        if(!build()){
            std::cout << "Failed building engine!" << std::endl;
        }
    }
}

YoloDetector::~YoloDetector(){
    // 销毁CUDA流
    cudaStreamDestroy(stream);

    for (int i{0}; i < 2; ++i)
    {
        CHECK(cudaFree(v_buffer_d[i]));
        v_buffer_d[i] = nullptr;
    }

    CHECK(cudaFree(transpose_device));
    transpose_device = nullptr;
    CHECK(cudaFree(decode_device));
    decode_device = nullptr;

    if(!output_data){
        delete [] output_data;
        output_data = nullptr;
    }

}

bool YoloDetector::build()
{
    auto start = std::chrono::high_resolution_clock::now();

    std::unique_ptr<nvinfer1::IBuilder> builder{std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger))};
    if(!builder) return false;
    std::unique_ptr<nvinfer1::INetworkDefinition> network{std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)))};
    if(!network) return false;
    
    auto profile= builder->createOptimizationProfile();
    if(!profile) return false;
    
    std::unique_ptr<nvinfer1::IBuilderConfig> config{std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig())};
    if(!config) return false;
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);
    std::unique_ptr<nvonnxparser::IParser> parser{std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_logger))};
    if(!parser) return false;
    
    // 量化模式
    nvinfer1::IInt8Calibrator* p_calibrator{nullptr};
    if (FP16_MODE){
        config->setFlag(BuilderFlag::kFP16);
    }

    if (INT8_MODE){
        config->setFlag(BuilderFlag::kINT8);
        int batch_size{8};
        p_calibrator = new Int8EntropyCalibrator2(batch_size, KInputW, KInputH, CalibrationDataPath.c_str(), CalibrationCacheFile.c_str());
        config->setInt8Calibrator(p_calibrator);
    }

    
    if (!parser->parseFromFile(ONNX_MODEL_PATH.c_str(), static_cast<int>(g_logger.reportableSeverity))){
        std::cout << std::string("Failed parsing .onnx file!") << std::endl;
        for (int i{0}; i < parser->getNbErrors(); ++i){
            auto *error = parser->getError(i);
            std::cout << std::to_string(static_cast<int>(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
        }
        return false;
    }
    std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

    nvinfer1::ITensor* input_tensor = network->getInput(0);
    profile->setDimensions(input_tensor->getName(), OptProfileSelector::kMIN, Dims{4, {1, 3, KInputH, KInputW}});
    profile->setDimensions(input_tensor->getName(), OptProfileSelector::kOPT, Dims{4, {1, 3, KInputH, KInputW}});
    profile->setDimensions(input_tensor->getName(), OptProfileSelector::kMAX, Dims{4, {1, 3, KInputH, KInputW}});
    config->addOptimizationProfile(profile);

    std::unique_ptr<nvinfer1::IHostMemory> engine_string {std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config))};
    std::cout << "Succeeded building serialized engine!" << std::endl;
    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_string->data(), engine_string->size()));
    if (engine == nullptr){ 
        std::cout << "Failed building engine!" << std::endl; 
        return false; 
    }
    std::cout << "Succeeded building engine!" << std::endl;
    std::ofstream engine_file(trt_file_, std::ios::binary);
    engine_file.write(static_cast<char *>(engine_string->data()), engine_string->size());
    std::cout << "engine file saved path: " << trt_file_ << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "YoloDetector::build::Building engine done! Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    return true;

}

std::vector<Detection> YoloDetector::inference(cv::Mat& img){
    if (img.empty()) return {};

    // 预处理
    preprocess(img, (float*)v_buffer_d[0], KInputH, KInputW, stream);

    // 模型推理
    context->enqueueV2(v_buffer_d.data(), stream, nullptr);

    // transpose [1 84 8400] convert to [1 8400 84]
    transpose((float*)v_buffer_d[1], transpose_device, output_candidates, num_class_ + 4, stream);
    
    // convert [1 8400 84] to [1 7001]
    decode(transpose_device, decode_device, output_candidates, num_class_, conf_thresh_, KMaxNumOutputBbox, KNumBoxElement, stream);
    
    // cuda nms
    nms(decode_device, nms_thresh_, KMaxNumOutputBbox, KNumBoxElement, stream);

    CHECK(cudaMemcpyAsync(output_data, decode_device, (1 + KMaxNumOutputBbox * KNumBoxElement) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    std::vector<Detection> results;
    int count = std::min((int)output_data[0], KMaxNumOutputBbox);
    for (int i {0}; i < count; i++){
        int pos  {1 + i * KNumBoxElement};
        int keepFlag = (int)output_data[pos + 6];
        if (keepFlag == 1){
            Detection det;
            memcpy(det.bbox, &output_data[pos], 4 * sizeof(float));
            det.conf = output_data[pos + 4];
            det.class_id = (int)output_data[pos + 5];
            results.emplace_back(det);
        }
    }

    for (size_t j {0}; j < results.size(); j++){
        scaleBbox(img, results[j].bbox);
    }
    return results;
}

void YoloDetector::drawImage(cv::Mat& img, std::vector<Detection>& infer_result){
    
    // 遍历检测结果
    for (size_t i{0}; i < infer_result.size(); i++){
        // 获取随机颜色
        cv::Scalar bbox_color(getRandomInt(), getRandomInt(), getRandomInt());
        // 获取检测框的坐标和大小
        cv::Rect r(
            round(infer_result[i].bbox[0]),
            round(infer_result[i].bbox[1]),
            round(infer_result[i].bbox[2] - infer_result[i].bbox[0]),
            round(infer_result[i].bbox[3] - infer_result[i].bbox[1])
        );
        // 在图像上绘制检测框
        cv::rectangle(img, r, bbox_color, 2);

        // 获取类别名称和置信度
        std::string class_name = ClassNames[(int)infer_result[i].class_id];
        std::string label_str = class_name + " " + std::to_string(infer_result[i].conf).substr(0, 4);

        // 获取文本大小
        cv::Size text_size = cv::getTextSize(label_str, cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
        // 获取文本框的坐标
        cv::Point top_left(r.x, r.y - text_size.height - 3);
        cv::Point bottom_right(r.x + text_size.width, r.y);
        // 在图像上绘制文本框
        cv::rectangle(img, top_left, bottom_right, bbox_color, -1);
        // 在图像上绘制文本
        cv::putText(img, label_str, cv::Point(r.x, r.y - 2), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
}
