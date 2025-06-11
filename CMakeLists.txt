cmake_minimum_required(VERSION 3.10)

project(yolo11_detect)

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE release)

find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("Embed_platform on")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
else()
  message("Embed_platform off")
  include_directories(/usr/include/x86_64-linux-gnu)
  link_directories(/usr/lib/x86_64-linux-gnu)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if(CUDA_FOUND)
    message("CUDA version: ${CUDA_VERSION}")
    message(STATUS "CUDA include files: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA library directories: ${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Cannot find CUDA")
endif()

if(TensorRT_FOUND)
    message(STATUS "Found TensorRT ${TensorRT_VERSION} in ${TensorRT_ROOT_DIR}")
    message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
    message(STATUS "TensorRT include path: ${TensorRT_INCLUDE_DIR}")
    include_directories( ${TensorRT_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Cannot find TensorRT")
endif()


if(OpenCV_FOUND)
    message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "Found OpenCV include folder: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "Found OpenCV libs: ${OpenCV_LIBS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Cannot find OpenCV")
endif()






include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(yolo_infer SHARED ${SRCS})


target_link_libraries(yolo_infer cudart nvinfer nvonnxparser ${OpenCV_LIBS})
# target_link_libraries(yolo_infer cudart ${TensorRT_LIBRARIES} ${OpenCV_LIBS} )

add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(main yolo_infer)