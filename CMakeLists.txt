cmake_minimum_required(VERSION 3.10)
project(parallel_thread_pool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开启所有警告并优化
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O0")

# 设置架构 (根据你的板子，RK3588 通常是 aarch64)
set(LIB_ARCH aarch64)

# ==========================================
# 1. 编译并行框架库 (thread_pool_lib)
# ==========================================
# 假设 parallel.h 是只有头文件的模板库，真正需要编译的是 thread_pool.cc
file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)
add_library(thread_pool_lib STATIC ${SRC_FILES})

# 设置头文件搜索路径
target_include_directories(thread_pool_lib PUBLIC 
    ${CMAKE_SOURCE_DIR}/src 
)

# ==========================================
# 2. 编译简单测试 (test_thread_pool)
# ==========================================
add_executable(test_thread_pool ${CMAKE_SOURCE_DIR}/example/test.cc)
target_link_libraries(test_thread_pool PRIVATE thread_pool_lib pthread)


# ==========================================
# 3. 编译 RKNN ResNet Demo (resnet18_thread)
# ==========================================

# --- 查找依赖库 ---
find_package(OpenCV REQUIRED)

# --- 设置第三方库路径 ---
# 设置 RGA 库
set(RGA_PATH ${CMAKE_SOURCE_DIR}/example_rknn/include/3rdparty/rga/RK3588)
# 注意：这里路径里的 // 是多余的，但通常 cmake 能容忍
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)

# 设置 RKNN Runtime 库 (!!! 关键修改 !!!)
# 请确认你的 librknnrt.so 在这个路径下，或者根据实际情况修改
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/example_rknn/lib)
set(RKNN_RT_LIB ${RKNN_API_PATH}/librknnrt.so) 
# 如果上面的路径不对，你可能需要手动指定绝对路径，例如:
# set(RKNN_RT_LIB /usr/lib/librknnrt.so)

# --- 设置头文件包含路径 ---
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src                      # 并行框架头文件
    ${CMAKE_SOURCE_DIR}/example_rknn/include     # 业务逻辑头文件
    ${CMAKE_SOURCE_DIR}/example_rknn/include/3rdparty # rknn_api.h 等
    ${RGA_PATH}/include                          # RgaApi.h 等
)

# --- 收集业务源代码 ---
# 必须包含 rkResnet 的实现文件 (例如 rknnPool.cpp, main_thread.cpp 等)
# 假设它们在 example_rknn/src 下
file(GLOB RKNN_SRC_FILES ${CMAKE_SOURCE_DIR}/example_rknn/src/*.cpp ${CMAKE_SOURCE_DIR}/example_rknn/src/*.cc)

# --- 生成可执行文件 ---
add_executable(resnet18_thread
    ${CMAKE_SOURCE_DIR}/example_rknn/main.cpp
    ${RKNN_SRC_FILES} 
)

# --- 链接库 ---
target_link_libraries(resnet18_thread
    thread_pool_lib   # 链接你的并行框架库
    ${OpenCV_LIBS}    # OpenCV
    ${RKNN_RT_LIB}    # RKNN Runtime
    ${RGA_LIB}        # RGA
    pthread           # 线程库
    dl                # 动态加载库 (rknnrt 通常需要)
)

# 输出路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})