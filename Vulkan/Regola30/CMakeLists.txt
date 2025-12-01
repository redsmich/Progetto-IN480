cmake_minimum_required(VERSION 3.10)
project(regola30)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

include_directories(${Vulkan_INCLUDE_DIRS} include)

add_executable(regola30 main.cpp)
target_link_libraries(regola30 ${Vulkan_LIBRARIES})

