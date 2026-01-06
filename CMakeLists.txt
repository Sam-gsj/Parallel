cmake_minimum_required(VERSION 3.10)

project(parallel_thread_pool)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")


file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)
add_library(thread_pool_lib STATIC ${SRC_FILES})

# 替换原有的 target_include_directories 行
target_include_directories(thread_pool_lib PUBLIC 
    ${CMAKE_SOURCE_DIR}/src  # 保留src目录
    ${CMAKE_SOURCE_DIR}      # 新增项目根目录
)


add_executable(test_thread_pool ${CMAKE_SOURCE_DIR}/example/test.cc)


target_link_libraries(test_thread_pool PRIVATE thread_pool_lib pthread)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})