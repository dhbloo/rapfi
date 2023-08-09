cmake_minimum_required(VERSION 3.18)

project(simde LANGUAGES CXX)

add_library(simde INTERFACE)

target_include_directories(simde INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
