cmake_minimum_required(VERSION 3.0.0)
project(yolov7)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(OpenCV REQUIRED)
set(NCNN_DIR "/home/ubuntu/DHY/ncnn-master")

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${NCNN_DIR}/src)
include_directories(${NCNN_DIR}/build/src)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${NCNN_DIR}/build/src)

add_executable(yolov7 ${PROJECT_SOURCE_DIR}/src/yolov7.cpp main.cpp)

target_link_libraries(yolov7 ${OpenCV_LIBS} ncnn pthread)
set_property(TARGET yolov7 PROPERTY CXX_STANDARD 11)
