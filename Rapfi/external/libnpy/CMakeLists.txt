cmake_minimum_required(VERSION 3.18)

project(libnpy LANGUAGES CXX)

add_library(libnpy INTERFACE)

target_include_directories(libnpy INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
