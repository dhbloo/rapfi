cmake_minimum_required(VERSION 3.18)

project(cpptoml LANGUAGES CXX)

add_library(cpptoml INTERFACE)

target_include_directories(cpptoml INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
