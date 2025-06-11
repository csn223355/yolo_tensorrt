#ifndef __CALIBRATOR_H_
#define __CALIBRATOR_H__

#include <string>
#include <vector>
#include <NvInfer.h>

using namespace nvinfer1;


/**
 * @class Int8EntropyCalibrator2
 * @brief 用于计算INT8量化所需的校准数据的类。
 *
 * Int8EntropyCalibrator2继承自IInt8EntropyCalibrator2接口，用于计算INT8量化所需的校准数据。
 * 该类通过读取指定目录下的图像文件，生成用于校准的batch数据，并支持读取和写入校准缓存。
 *
 * 核心功能包括：
 * - 获取batch size
 * - 获取batch数据
 * - 读取校准缓存
 * - 写入校准缓存
 *
 * 使用示例：
 *
 * 构造函数参数：
 * - int batch_size: 每个batch中的图像数量。
 * - int input_w: 输入图像的宽度。
 * - int input_h: 输入图像的高度。
 * - const char* img_dir: 包含校准图像的目录路径。
 * - const char* calib_table_name: 校准表的名称。
 * - bool read_cache: 是否读取已有的校准缓存，默认为true。
 *
 * 特殊使用限制或潜在的副作用：
 * - 确保提供的图像目录路径有效且包含足够的图像文件。
 * - 校准缓存的大小和内容应与校准过程的要求相匹配。
 */
class Int8EntropyCalibrator2 : public IInt8EntropyCalibrator2
{
public:
    /**
     * @brief 构造函数，初始化Int8EntropyCalibrator2对象
     * 
     * @param batch_size 批次大小
     * @param input_w 输入图像的宽度
     * @param input_h 输入图像的高度
     * @param img_dir 图像目录路径
     * @param calib_table_name 校准表名称
     * @param read_cache 是否读取缓存，默认为true
     */
    Int8EntropyCalibrator2(int batch_size, int input_w, int input_h, const char* img_dir, const char* calib_table_name, bool read_cache=true);

    /**
     * @brief 析构函数，释放Int8EntropyCalibrator2对象
     */
    virtual ~Int8EntropyCalibrator2();

    /**
     * @brief 获取batch size
     * 
     * @return int 返回批次大小
     */
    int getBatchSize() const noexcept override;

    /**
     * @brief 获取batch数据
     * 
     * @param bindings 绑定数据指针数组
     * @param names 绑定名称数组
     * @param nbBindings 绑定数量
     * @return bool 是否成功获取batch数据
     */
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    /**
     * @brief 读取校准缓存
     * 
     * @param length 缓存长度
     * @return const void* 返回校准缓存指针
     */
    const void* readCalibrationCache(size_t& length) noexcept override;

    /**
     * @brief 写入校准缓存
     * 
     * @param cache 缓存数据指针
     * @param length 缓存长度
     */
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    // batch size
    int batch_size_;
    // 输入图像宽度
    int input_w_;
    // 输入图像高度
    int input_h_;
    // 当前图像索引
    int img_idx_;
    // 图像目录路径
    std::string img_dir_;
    // 图像文件列表
    std::vector<std::string> img_files_;
    // 输入计数
    size_t input_count_;
    // 校准表名称
    std::string calib_table_name_;
    // 是否读取缓存
    bool read_cache_;
    // batch数据指针
    float* batch_data;
    // 设备输入指针
    void* device_input_;
    // 校准缓存
    std::vector<char> calib_cache_;
};

#endif  // __CALIBRATOR_H__
