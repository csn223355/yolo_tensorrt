#include "preprocess.h"


__global__ void letterbox(const uchar* src_data, const int src_h, const int src_w, 
    uchar* tgt_data, const int tgt_h, const int tgt_w, 
    const int rsz_h, const int rsz_w, const int start_y, const int start_x)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x; //  计算当前线程的x和y坐标
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * tgt_w; //  计算当前线程在目标图像中的索引
    int idx3 = idx * 3; //  计算当前线程在目标图像中的索引，乘以3，因为每个像素有3个通道

    if ( ix > tgt_w || iy > tgt_h ) return;   //  如果当前线程的x坐标大于目标图像的宽度，或者y坐标大于目标图像的高度，则返回
    
    if ( iy < start_y || iy > (start_y + rsz_h - 1) ) { //  如果 iy 小于 start_y 或者大于 start_y + rsz_h - 1，则将 tgt_data[idx3] 到 tgt_data[idx3 + 2] 的值设为 128，并返回
        tgt_data[idx3] = 128;
        tgt_data[idx3 + 1] = 128;
        tgt_data[idx3 + 2] = 128;
        return;
    }
    if ( ix < start_x || ix > (start_x + rsz_w - 1) ){ //  如果 ix 小于 start_x 或者大于 start_x + rsz_w - 1，则将 tgt_data[idx3] 到 tgt_data[idx3 + 2] 的值设为 128，并返回
        tgt_data[idx3] = 128;
        tgt_data[idx3 + 1] = 128;
        tgt_data[idx3 + 2] = 128;
        return;
    }

    float scale_y = (float)rsz_h / (float)src_h; //  计算缩放比例
    float scale_x = (float)rsz_w / (float)src_w;

    // (ix,iy)为目标图像坐标
    // (before_x,before_y)原图坐标
    float before_x = float(ix - start_x + 0.5) / scale_x - 0.5;
    float before_y = float(iy - start_y + 0.5) / scale_y - 0.5;
    // 原图像坐标四个相邻点
    // 获得变换前最近的四个顶点,取整
    int top_y = static_cast<int>(before_y);
    int bottom_y = top_y + 1;
    int left_x = static_cast<int>(before_x);
    int right_x = left_x + 1;
    //计算变换前坐标的小数部分
    float u = before_x - left_x;
    float v = before_y - top_y;

    if (top_y >= src_h - 1 && left_x >= src_w - 1)  //右下角
    {
        for (int k = 0; k < 3; k++)
        {
            tgt_data[idx3 + k] = (1. - u) * (1. - v) * src_data[(left_x + top_y * src_w) * 3 + k];
        }
    }
    else if (top_y >= src_h - 1)  // 最后一行
    {
        for (int k = 0; k < 3; k++)
        {
            tgt_data[idx3 + k]
            = (1. - u) * (1. - v) * src_data[(left_x + top_y * src_w) * 3 + k]
            + (u) * (1. - v) * src_data[(right_x + top_y * src_w) * 3 + k];
        }
    }
    else if (left_x >= src_w - 1)  // 最后一列
    {
        for (int k = 0; k < 3; k++)
        {
            tgt_data[idx3 + k]
            = (1. - u) * (1. - v) * src_data[(left_x + top_y * src_w) * 3 + k]
            + (1. - u) * (v) * src_data[(left_x + bottom_y * src_w) * 3 + k];
        }
    }
    else  // 非最后一行或最后一列情况
    {
        for (int k = 0; k < 3; k++)
        {
            tgt_data[idx3 + k]
            = (1. - u) * (1. - v) * src_data[(left_x + top_y * src_w) * 3 + k]
            + (u) * (1. - v) * src_data[(right_x + top_y * src_w) * 3 + k]
            + (1. - u) * (v) * src_data[(left_x + bottom_y * src_w) * 3 + k]
            + u * v * src_data[(right_x + bottom_y * src_w) * 3 + k];
        }
    }
}

__global__ void process(const uchar* src_data, float* tgt_data, const int h, const int w)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * w;
    int idx3 = idx * 3;

    if (ix < w && iy < h) //  如果当前线程的x坐标小于图像的宽度，且y坐标小于图像的高度
    {
        tgt_data[idx] = (float)src_data[idx3 + 2] / 255.0;   //  将源图像的R通道像素值除以255.0，存入目标图像的R通道
        tgt_data[idx + h * w] = (float)src_data[idx3 + 1] / 255.0;   //  将源图像的G通道像素值除以255.0，存入目标图像的G通道
        tgt_data[idx + h * w * 2] = (float)src_data[idx3] / 255.0;   //  将源图像的B通道像素值除以255.0，存入目标图像的B通道
    }
}

void preprocess(const cv::Mat& src, float* dst_dev_data, const int dst_height, const int dst_width, cudaStream_t stream)
{
    int src_height {src.rows}; //  获取源图像的高度和宽度
    int src_width {src.cols};
    int src_elements {src_height * src_width * 3}; //  计算源图像的元素个数
    int dst_elements {dst_height * dst_width * 3}; //  计算目标图像的元素个数

    // middle image data on device ( for bilinear resize )
    uchar* mid_dev_data;
    cudaMalloc((void**)&mid_dev_data, sizeof(uchar) * dst_elements);
    // source images data on device
    uchar* src_dev_data;
    cudaMalloc((void**)&src_dev_data, sizeof(uchar) * src_elements);
    cudaMemcpyAsync(src_dev_data, src.data, sizeof(uchar) * src_elements, cudaMemcpyHostToDevice, stream);

    // calculate width and height after resize
    int w, h, x, y;
    float r_w {dst_width / (src_width * 1.0)}; //  计算缩放后的宽度和高度
    float r_h {dst_height / (src_height * 1.0)};
    if (r_h > r_w) {
        w = dst_width; //  如果高度缩放比例大于宽度缩放比例，则宽度保持不变，高度按宽度缩放比例缩放
        h = r_w * src_height;
        x = 0;
        y = (dst_height - h) / 2;
    }
    else {
        w = r_h * src_width; //  如果宽度缩放比例大于高度缩放比例，则高度保持不变，宽度按高度缩放比例缩放
        h = dst_height;
        x = (dst_width - w) / 2;
        y = 0;
    }
    
    dim3 block_size(32, 32); //  定义block和grid的大小
    dim3 grid_size((dst_width + block_size.x - 1) / block_size.x, (dst_height + block_size.y - 1) / block_size.y);

    // letterbox and resize
    letterbox<<<grid_size, block_size, 0, stream>>>(src_dev_data, src_height, src_width, mid_dev_data, dst_height, dst_width, h, w, y, x);
    cudaDeviceSynchronize();
    // hwc to chw / bgr to rgb / normalize
    process<<<grid_size, block_size, 0, stream>>>(mid_dev_data, dst_dev_data, dst_height, dst_width);

    cudaFree(src_dev_data); //  释放内存
    cudaFree(mid_dev_data);
}
