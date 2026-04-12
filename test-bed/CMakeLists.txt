cmake_minimum_required(VERSION 3.20)
project(test_bed LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90 120)

find_package(CUDAToolkit REQUIRED)

# Bank conflict test kernels
set(KERNELS
    kernels/01_baseline_r1y.cu
    kernels/02_transposed_layout.cu
    kernels/03_32x32_tiles.cu
    kernels/04_128x64_tiles.cu
    kernels/05_warp_tiled.cu
)

# Build each kernel as separate executable
foreach(kernel ${KERNELS})
    get_filename_component(name ${kernel} NAME_WE)
    add_executable(${name} ${kernel})
    target_link_libraries(${name} PRIVATE CUDA::cudart)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endforeach()

# Install targets to build/ folder
foreach(kernel ${KERNELS})
    get_filename_component(name ${kernel} NAME_WE)
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:${name}>
            ${CMAKE_BINARY_DIR}/${name}
    )
endforeach()