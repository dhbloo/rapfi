cmake_minimum_required(VERSION 3.18)

project(BS_thread_pool LANGUAGES CXX)

add_library(BS_thread_pool INTERFACE)

target_include_directories(BS_thread_pool INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
