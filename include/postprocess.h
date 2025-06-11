#ifndef __POSTPROCESS_H__
#define __POSTPROCESS_H__


#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "config.h"

/**
* @brief transpose [1 84 8400] convert to [1 8400 84]
* @param src Tensor, dim is [1 84 8400]
* @param dst Tensor, dim is [1 8400 84]
* @param num_bboxes number of bboxes
* @param num_elements center_x, center_y, width, height, 80 or other classes
*/
void transpose(float* src, float* dst, int num_bboxes, int num_elements, cudaStream_t stream);


/**
* @brief convert [1 8400 84] to [1 7001](7001 = 1 + 1000 * 7, 1: number of valid bboxes
     1000: max bboxes, valid bboxes may less than 1000, 7: left, top, right, bottom, confidence, class, keepflag)
* @param src Tensor, dim is [1 8400 84]
* @param dst Tensor, dim is [1 7001]
* @param num_bboxes number of bboxes
* @param num_classes number of classes
* @param conf_thresh confidence threshold
* @param max_bjects max objects
* @param num_box_element number of box elements
* @param stream cuda stream
*/
void decode(float* src, float* dst, int num_bboxes, int num_classes, float conf_thresh, int max_bjects, int num_box_element, cudaStream_t stream);


/**
* @brief 非极大值抑制（NMS）
* @param data Tensor, dim is [1 7001]
* @param k_nms_thresh nms threshold
* @param max_objects    max objects
* @param num_box_element number of box elements
* @param stream cuda stream
*/
void nms(float* data, float k_nms_thresh, int max_objects, int num_box_element, cudaStream_t stream);


/***
 * @brief scaleBbox
 * @details 将bbox的坐标从归一化坐标转换为原始图像坐标
 * @param img 原始图像
 * @param bbox bbox的归一化坐标
*/
__inline__ void scaleBbox(cv::Mat& img, float bbox[4])
{
    float r_w {KInputW / (img.cols * 1.0)};
    float r_h {KInputH / (img.rows * 1.0)};
    float r {std::min(r_w, r_h)};
    float pad_h {(KInputH - r * img.rows) / 2};
    float pad_w {(KInputW - r * img.cols) / 2};

    bbox[0] = (bbox[0] - pad_w) / r;
    bbox[1] = (bbox[1] - pad_h) / r;
    bbox[2] = (bbox[2] - pad_w) / r;
    bbox[3] = (bbox[3] - pad_h) / r;
}



















#endif /* __POSTPROCESS_H__ */