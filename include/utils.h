#ifndef __UTILS_H__
#define __UTILS_H__

#include <dirent.h>
#include <random>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <vector>
#include <iostream>
#include <filesystem> // C++17 引入


/**
 * @brief 读取指定目录中的文件名，并将其存储在提供的向量中。
 *
 * 该函数接收一个目录名和一个字符串向量作为参数，遍历指定目录中的所有文件和子目录，
 * 将非"."和".."的文件名添加到向量中。如果目录打开失败，函数返回-1；否则返回0。
 *
 * @param p_dir_name 指向目录名的C风格字符串。
 * @param file_names 用于存储文件名的字符串向量。
 * @return 成功时返回0，失败时返回-1。
 */
static inline int readFilesInDir(const char* p_dir_name, std::vector<std::string>& file_names)
{
    // 尝试打开指定目录
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        // 如果目录打开失败，返回-1
        return -1;
    }

    // 初始化指向目录项的指针
    struct dirent* p_file = nullptr;
    // 遍历目录中的所有项
    while ((p_file = readdir(p_dir)) != nullptr) {
        // 排除"."和".."目录项
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            // 将文件名转换为std::string并添加到向量中
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    // 关闭目录
    closedir(p_dir);
    // 返回0表示成功
    return 0;
}


/**
 * @brief 获取处于某一范围内的一个随机整数
 *
 * 该函数使用C++标准库中的随机数生成器来生成一个指定范围内的随机整数。
 * 默认情况下，随机整数的范围是从0到255。
 *
 * @param min_thres 随机整数的最小值（包含在内），默认值为0
 * @param max_thres 随机整数的最大值（包含在内），默认值为255
 * @return 返回生成的随机整数
 */
static inline int getRandomInt(int min_thres=0, int max_thres=255){
    // 获取处于某一范围内的一个随机整数
    std::random_device rd; // 用于初始化随机数生成器的种子
    std::mt19937 gen(rd()); // 使用Mersenne Twister 19937算法的随机数生成器
    std::uniform_int_distribution<> distrib(min_thres, max_thres); // 均匀分布的整数范围

    int random_integer = distrib(gen); // 生成随机整数

    return random_integer; // 返回生成的随机整数
}

/**
 * @brief 读取指定路径的引擎文件，并将其内容存储到提供的向量中。
 *
 * 该函数首先检查指定的引擎文件是否存在。如果文件存在，则尝试打开文件并读取其内容。
 * 文件内容将被存储到传入的向量中。如果在任何步骤中发生错误，函数将输出相应的错误信息。
 *
 * @param engine_path 引擎文件的路径。
 * @param engine_data 用于存储引擎文件内容的向量。
 * @return 返回读取的文件大小，如果发生错误则返回 0。
 */
inline size_t readEngine(const std::string& engine_path, std::vector<char>& engine_data) 
{
    
    if (std::filesystem::exists(engine_path)) {
        std::ifstream file(engine_path, std::ios::binary);
        
        // 检查文件是否成功打开
        if (!file.is_open()) {
            std::cerr << "Failed to open engine file!" << std::endl;
            return 0;
        }

        // 获取文件大小
        file.seekg(0, std::ios::end);
        std::streamsize fsize = file.tellg();
        if (fsize == -1) {
            std::cerr << "Failed to get file size!" << std::endl;
            return 0;
        }
        file.seekg(0, std::ios::beg);

        // 读取文件内容
        engine_data.clear();
        engine_data.resize(fsize);
        if (!file.read(engine_data.data(), fsize)) {
            std::cerr << "Failed reading engine file!" << std::endl;
            return 0;
        }

        std::cout << "Succeeded getting serialized engine!" << std::endl;
        return static_cast<size_t>(fsize);

    } else {
        std::cerr << "Engine file does not exist!" << std::endl;
        return 0;
    }
}






#endif  // __UTILS_H__
