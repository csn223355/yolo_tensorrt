#ifndef __INFER_H__
#define __INFER_H__

#include <opencv2/opencv.hpp>

#include "public.h"
#include "config.h"
#include "types.h"

using namespace nvinfer1;


/**
 * @brief YOLO目标检测器类，用于加载YOLO模型并进行目标检测。
 *
 * 该类封装了YOLO目标检测的核心功能，包括模型加载、推理和结果绘制。
 * 用户可以通过构造函数指定模型文件路径、GPU ID、NMS阈值、置信度阈值和类别数量。
 * 主要方法包括：
 * - inference：对输入图像进行推理，返回检测到的目标列表。
 * - drawImage：在图像上绘制检测到的目标。
 *
 * 示例：
 *
 * @param trt_file YOLO模型的TensorRT文件路径。
 * @param gpu_id 使用的GPU ID，默认值为KGpuId。
 * @param nms_thresh 非极大值抑制阈值，默认值为KNmsThresh。
 * @param conf_thresh 置信度阈值，默认值为KConfThresh。
 * @param num_class 类别数量，默认值为KNumClass。
 */
class YoloDetector
{
public:
    /**
     * @brief 构造函数，初始化YOLO检测器。
     * @param trt_file TensorRT引擎文件路径。
     * @param gpu_id 使用的GPU设备ID，默认值为KGpuId。
     * @param nms_thresh 非极大值抑制阈值，默认值为KNmsThresh。
     * @param conf_thresh 置信度阈值，默认值为KConfThresh。
     * @param num_class 类别数量，默认值为KNumClass。
     */
    YoloDetector(
        const std::string trt_file,
        int gpu_id=KGpuId,
        float nms_thresh=KNmsThresh,
        float conf_thresh=KConfThresh,
        int num_class=KNumClass
    );

    /**
     * @brief 析构函数，释放YOLO检测器相关资源。
     */
    ~YoloDetector();

    /**
     * @brief 进行目标检测推理。
     * @param img 输入图像，使用OpenCV的Mat格式。
     * @return 检测结果，包含检测到的目标信息。
     */
    std::vector<Detection> inference(cv::Mat& img);

    /**
     * @brief 在图像上绘制检测到的目标。
     * @param img 输入图像，使用OpenCV的Mat格式。
     * @param infer_result 检测结果，包含检测到的目标信息。
     */
    static void drawImage(cv::Mat& img, std::vector<Detection>& infer_result);

private:
    /**
     * @brief 获取TensorRT引擎。
     */
    void getEngine();

private:
    Logger              g_logger;           /**< 日志记录器。 */
    std::string         trt_file_;          /**< TensorRT引擎文件路径。 */

    int                 num_class_;         /**< 类别数量。 */
    float               nms_thresh_;        /**< 非极大值抑制阈值。 */
    float               conf_thresh_;       /**< 置信度阈值。 */

    ICudaEngine *       engine;             /**< TensorRT引擎。 */
    IRuntime *          runtime;            /**< TensorRT运行时。 */
    IExecutionContext * context;            /**< TensorRT执行上下文。 */

    cudaStream_t        stream;             /**< CUDA流。 */

    float *             output_data;        /**< 输出数据。 */
    std::vector<void *> v_buffer_d;         /**< 设备端缓冲区。 */
    float *             transpose_device;   /**< 转置后的设备端数据。 */
    float *             decode_device;      /**< 解码后的设备端数据。 */

    int                 OUTPUT_CANDIDATES;  /**< 输出候选框数量，默认值为8400。 */
};          // class YoloDetector

#endif  // __INFER_H__
