#ifndef __PUBLIC_H__
#define __PUBLIC_H__

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string.h>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <map>

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#define CHECK(call) check(call, __LINE__, __FILE__)



/**
 * @brief 检查CUDA运行时API调用的返回值，并在出错时输出错误信息。
 *
 * 该函数用于检查CUDA运行时API调用的返回值。如果返回值表示错误（即不等于cudaSuccess），
 * 则输出错误信息，包括错误名称、错误发生的代码行号和文件名。否则，返回true表示调用成功。
 *
 * @param e CUDA运行时API调用的返回值（cudaError_t类型）
 * @param iLine 错误发生的代码行号（int类型）
 * @param szFile 错误发生的文件名（const char*类型）
 * @return 如果返回值表示错误，则返回false；否则返回true
 */
inline bool check(cudaError_t e, int iLine, const char *szFile)
{
    // 如果错误码不等于cudaSuccess，则输出错误信息
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    // 否则返回true
    return true;
}

using namespace nvinfer1;


class Logger : public ILogger
{
public:
    /**
     * @brief 定义可报告的严重程度
     * 
     * 该成员变量用于存储当前Logger实例可报告的最小日志严重程度。
     * 只有当日志的严重程度大于或等于该值时，日志才会被输出。
     */
    Severity reportableSeverity;

    /**
     * @brief 构造函数，设置可报告的严重程度
     * 
     * 构造Logger实例时，可以指定一个初始的可报告严重程度。
     * 如果未指定，则默认为Severity::kINFO。
     * 
     * @param severity 可报告的严重程度，默认为Severity::kINFO
     */
    Logger(Severity severity = Severity::kINFO):
        reportableSeverity(severity) {}

    /**
     * @brief 重写log函数，根据严重程度输出日志信息
     * 
     * 该函数根据传入的日志严重程度和消息内容输出日志信息。
     * 如果传入的严重程度大于可报告的严重程度，则不输出日志。
     * 否则，根据严重程度输出相应的日志前缀和消息内容。
     * 
     * @param severity 日志的严重程度
     * @param msg 日志消息内容
     * @throws noexcept 保证该函数不会抛出异常
     */
    void log(Severity severity, const char *msg) noexcept override
    {
        // 如果严重程度大于可报告的严重程度，则不输出日志
        if (severity > reportableSeverity)
        {
            return;
        }
        // 根据严重程度输出日志信息
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

/**
 * @brief 根据传入的数据类型，返回对应的大小
 * 
 * 该函数接收一个枚举类型 DataType 的参数，并根据该参数的值返回相应数据类型的大小。
 * 
 * @param dataType 数据类型枚举值
 * @return size_t 对应数据类型的大小
 */
__inline__ size_t dataTypeToSize(DataType dataType)
{
    
    switch ((int)dataType)
    {
        case int(DataType::kFLOAT):
            return 4;
        case int(DataType::kHALF):
            return 2;
        case int(DataType::kINT8):
            return 1;
        case int(DataType::kINT32):
            return 4;
        case int(DataType::kBOOL):
            return 1;
        default:
            return 4;
    }
}

/**
 * @brief 将Dims32类型的dim转换为字符串类型
 * @param dim Dims32类型的dim
 * @return std::string 字符串类型的dim
*/
__inline__ std::string shapeToString(Dims32 dim)
{
    // 将Dims32类型的dim转换为字符串类型
    std::string output("(");
    // 如果dim的维度为0，则返回空字符串
    if (dim.nbDims == 0)
    {
        return output + std::string(")");
    }
    // 遍历dim的维度，将每个维度转换为字符串并添加到output中
    for (int i = 0; i < dim.nbDims - 1; i++)
    {
        output += std::to_string(dim.d[i]) + std::string(", ");
    }
    // 将最后一个维度转换为字符串并添加到output中
    output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
    return output;
}


/**
 * @brief 根据传入的数据类型，返回相应的字符串
 * @param dataType 数据类型枚举值
 * @return std::string 对应数据类型的字符串
*/
__inline__ std::string dataTypeToString(DataType dataType)
{
    // 根据传入的数据类型，返回相应的字符串
    switch (dataType)
    {
    case DataType::kFLOAT:
        // 如果数据类型为kFLOAT，返回"FP32 "
        return std::string("FP32 ");
    case DataType::kHALF:
        // 如果数据类型为kHALF，返回"FP16 "
        return std::string("FP16 ");
    case DataType::kINT8:
        // 如果数据类型为kINT8，返回"INT8 "
        return std::string("INT8 ");
    case DataType::kINT32:
        // 如果数据类型为kINT32，返回"INT32"
        return std::string("INT32");
    case DataType::kBOOL:
        // 如果数据类型为kBOOL，返回"BOOL "
        return std::string("BOOL ");
    default:
        // 如果数据类型未知，返回"Unknown"
        return std::string("Unknown");
    }
}

#endif  // __PUBLIC_H__
