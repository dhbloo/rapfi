cmake_minimum_required(VERSION 3.18)

project(cxxopts LANGUAGES CXX)

add_library(cxxopts INTERFACE)

target_include_directories(cxxopts INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
