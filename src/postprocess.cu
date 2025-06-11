#include "postprocess.h"



// 转置核函数
__global__ void transposeKernel(float* src, float* dst, int num_bboxes, int num_elements, int edge)
{
    // 计算线程位置
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    // 如果位置超出范围，则返回
    if (position >= edge) return;

    // 转置操作
    dst[position] = src[(position % num_elements) * num_bboxes + position / num_elements];
}


// 转置函数
void transpose(float* src, float* dst, int num_bboxes, int num_elements, cudaStream_t stream)
{
    // 计算转置后的边长
    int edge = num_bboxes * num_elements;
    // 定义块大小和网格大小
    int block_size {256};
    int grid_size {(edge + block_size - 1) / block_size};
    // 调用转置函数
    transposeKernel<<<grid_size, block_size, 0, stream>>>(src, dst, num_bboxes, num_elements, edge);
}



// 解码核函数
__global__ void decodeKernel(float* src, float* dst, int num_bboxes, int num_classes, float conf_thresh, int max_bjects, int num_box_elementnt){
    // 计算线程位置
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    // 如果位置超出范围，则返回
    if (position >= num_bboxes) return;

    // 获取当前bbox的指针
    float* pitem = src + (4 + num_classes) * position;
    // 获取当前bbox的类别置信度
    float* classConf = pitem + 4;
    // 初始化置信度和类别
    float confidence = 0;
    int label = 0;
    // 遍历所有类别，找到置信度最高的类别
    for (int i = 0; i < num_classes; i++){
        if (classConf[i] > confidence){
            confidence = classConf[i];
            label = i;
        }
    }

    // 如果置信度低于阈值，则返回
    if (confidence < conf_thresh) return;

    // 获取当前bbox的索引
    int index = (int)atomicAdd(dst, 1);
    // 如果索引超出范围，则返回
    if (index >= max_bjects) return;

    // 获取当前bbox的中心点、宽度和高度
    float cx     = pitem[0];
    float cy     = pitem[1];
    float width  = pitem[2];
    float height = pitem[3];

    // 计算当前bbox的左上角和右下角坐标
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    // 获取输出bbox的指针
    float* pout_item = dst + 1 + index * num_box_elementnt;
    // 将当前bbox的信息写入输出bbox
    pout_item[0] = left;
    pout_item[1] = top;
    pout_item[2] = right;
    pout_item[3] = bottom;
    pout_item[4] = confidence;
    pout_item[5] = label;
    pout_item[6] = 1;  // 1 = keep, 0 = ignore
}


// decode函数
void decode(float* src, float* dst, int num_bboxes, int num_classes, float conf_thresh, int max_bjects, int num_box_elementnt, cudaStream_t stream){
    // 将输出bbox的计数器置为0
    cudaMemsetAsync(dst, 0, sizeof(int), stream);
    // 定义块大小和网格大小
    int block_size {256};
    int grid_size {(num_bboxes + block_size - 1) / block_size};
    // 调用解码函数
    decodeKernel<<<grid_size, block_size, 0, stream>>>(src, dst, num_bboxes, num_classes, conf_thresh, max_bjects, num_box_elementnt);
}


// ------------------ nms --------------------
// 计算两个bbox的iou
__device__ float boxIou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom)
{
    // 计算两个bbox的交集的左上角和右下角坐标
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    // 计算交集的面积
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    // 如果交集面积为0，则iou为0
    if (c_area == 0.0f) return 0.0f;

    // 计算两个bbox的面积
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    // 计算iou
    return c_area / (a_area + b_area - c_area);
}


// nms核函数
__global__ void nmsKernel(float* data, float k_nms_thresh, int max_bjects, int num_box_elementnt){
    // 计算线程位置
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    // 获取bbox的总数
    int count = min((int)data[0], max_bjects);
    // 如果位置超出范围，则返回
    if (position >= count) return;

    // left, top, right, bottom, confidence, class, keepflag
    // 获取当前bbox的指针
    float* pcurrent = data + 1 + position * num_box_elementnt;
    // 遍历所有bbox
    float* pitem;
    for (int i = 0; i < count; i++){
        // 获取当前bbox的指针
        pitem = data + 1 + i * num_box_elementnt;
        // 如果当前bbox和遍历到的bbox是同一个，或者类别不同，则跳过
        if (i == position || pcurrent[5] != pitem[5]) continue;

        // 如果遍历到的bbox的置信度大于等于当前bbox的置信度
        if (pitem[4] >= pcurrent[4]){
            // 如果置信度相同，且遍历到的bbox的索引小于当前bbox的索引，则跳过
            if (pitem[4] == pcurrent[4] && i < position) continue;

            // 计算两个bbox的iou
            float iou = boxIou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]
            );

            // 如果iou大于阈值，则将当前bbox的keepflag置为0
            if (iou > k_nms_thresh){
                pcurrent[6] = 0;  // 1 = keep, 0 = ignore
                return;
            }
        }
    }
}


// 调用nms函数
void nms(float* data, float k_nms_thresh, int max_bjects, int num_box_elementnt, cudaStream_t stream){
    // 定义块大小和网格大小
    int block_size = max_bjects < 256?max_bjects:256;
    int grid_size = (max_bjects + block_size - 1) / block_size;
    // 调用nms函数
    nmsKernel<<<grid_size, block_size, 0, stream>>>(data, k_nms_thresh, max_bjects, num_box_elementnt);
}
