cmake_minimum_required(VERSION 3.18)
project(voxel_rt CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(voxel_rt main.cu)

target_compile_options(voxel_rt PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
