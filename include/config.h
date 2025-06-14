#ifndef __CONFIG_H__
#define __CONFIG_H__

/**
 * @file      config.h
 * @brief     
 *
 * Copyright (c) 2024 
 *
 * @author    shiningchen
 * @date      2025.06.14
 * @version   1.0
*/


#include <iostream>
#include <string>
#include <vector>

constexpr int  KGpuId {0};
constexpr int KNumClass {9};
constexpr int KInputH {1280};
constexpr int KInputW {1280};
constexpr float KNmsThresh {0.45f};
constexpr float KConfThresh {0.5f};
constexpr int KMaxNumOutputBbox {1000};  // assume the box outputs no more than kMaxNumOutputBbox boxes that conf >= kNmsThresh;
constexpr int KNumBoxElement {7};  // left, top, right, bottom, confidence, class, keepflag(whether drop when NMS)

const std::string ONNX_MODEL_PATH {"../model_weights/onnx/best.onnx"};

constexpr bool FP16_MODE {false};
constexpr bool INT8_MODE {false};


const std::string CalibrationCacheFile {"./int8.cache"};
const std::string CalibrationDataPath {"../calibrator"};  // 存放用于 int8 量化校准的图像

const std::vector<std::string> ClassNames {
    "microscope", "flow_cytometry", "WB", "mouse", "cell_migration", "cell_clone", "fluorescence", "electron_microscopy", "organization"
}; // 每个类别对应的名称



#endif /* __CONFIG_H__ */