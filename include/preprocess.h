#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

/**
* @param src: input image 
* @param dst_dev_data: device memory to store preprocessed image data
* @param dst_height: output image height, CNN input height
* @param dst_width: output image width, CNN input width
* @param stream: cuda stream
*/
void preprocess(const cv::Mat& src, float* dst_dev_data, const int dst_height, const int dst_width, cudaStream_t stream);










#endif /* __PREPROCESS_H__ */