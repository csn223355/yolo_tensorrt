# 编译与运行
## 环境依赖
`cuda-11.8`, `cudnn:8.5`, `TensorRT:8.6`, `opencv:4.5.3`
## 编译脚本
```bash
git clone https://github.com/csn223355/yolo_tensorrt.git
cd yolo_tensorrt
cmake -S . -B build
cmake --build build -j6
```
## 运行测试程序
```bash
./build/main ../test_images
```

# TODOList
1. - [] 前后处理进一步集成到类中
2. - [] 抽象出推理基类
3. - [] 增加yolov11分割模型推理
