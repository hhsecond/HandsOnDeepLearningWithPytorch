cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(fizbuz fizbuz.cpp)
target_link_libraries(fizbuz "${TORCH_LIBRARIES}")
set_property(TARGET fizbuz PROPERTY CXX_STANDARD 11)