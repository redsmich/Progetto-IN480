cmake_minimum_required(VERSION 3.10)
project(gaussian_blur)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

include_directories(${Vulkan_INCLUDE_DIRS} include)

add_executable(gaussian_blur main.cpp)
target_link_libraries(gaussian_blur ${Vulkan_LIBRARIES})

