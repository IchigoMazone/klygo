cmake_minimum_required(VERSION 3.15)

project(klygo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MINGW)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -static-libstdc++ -Wl,-Bstatic -lgomp -lpthread -Wl,-Bdynamic")
endif()

# Library
add_library(klygo_core
    src/cpu_allocator.cpp
    src/storage.cpp
    src/tensor_impl.cpp
    src/tensor.cpp
    src/tensor_factory.cpp
    src/tensor_printer.cpp
)

target_include_directories(klygo_core PUBLIC ${PROJECT_SOURCE_DIR}/include)

# OpenMP Configuration
# macOS Apple Clang does not ship with OpenMP — find Homebrew libomp
if(APPLE)
    # Try common Homebrew prefix (Intel and Apple Silicon)
    foreach(_prefix "/usr/local" "/opt/homebrew")
        if(EXISTS "${_prefix}/opt/libomp/include/omp.h")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_prefix}/opt/libomp/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${_prefix}/opt/libomp/lib/libomp.dylib")
        endif()
    endforeach()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(klygo_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# CUDA Configuration (Compatible across CUDA 10.x, 11.x, 12.x+)
find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
    find_package(CUDA QUIET)
endif()

if(CUDAToolkit_FOUND OR CUDA_FOUND)
    message(STATUS "CUDA found. Enabling GPU support across GPU architectures.")
    target_compile_definitions(klygo_core PUBLIC KLYGO_USE_CUDA)
    
    if(CUDA_FOUND AND NOT CUDAToolkit_FOUND)
        cuda_add_library(klygo_cuda src/cuda_kernels.cu)
        target_link_libraries(klygo_core PUBLIC klygo_cuda)
    else()
        enable_language(CUDA)
        set(CMAKE_CUDA_ARCHITECTURES "all-major")
        target_sources(klygo_core PRIVATE src/cuda_kernels.cu)
    endif()
else()
    message(STATUS "CUDA NOT found. Compiling CPU-only build.")
endif()

# Wrapper C++
find_package(pybind11 REQUIRED)
pybind11_add_module(klygo python/binding.cpp)
target_link_libraries(klygo PRIVATE klygo_core)

# If CUDA is found, compile klygo module with CUDA flag as well
if(CUDA_FOUND)
    target_compile_definitions(klygo PRIVATE KLYGO_USE_CUDA)
endif()

# Test
add_executable(tensor_01 test/tensor_01.cpp)
target_link_libraries(tensor_01 PRIVATE klygo_core)

add_executable(tensor_02 test/tensor_02.cpp)
target_link_libraries(tensor_02 PRIVATE klygo_core)

add_executable(tensor_03 test/tensor_03.cpp)
target_link_libraries(tensor_03 PRIVATE klygo_core)

add_executable(tensor_04 test/tensor_04.cpp)
target_link_libraries(tensor_04 PRIVATE klygo_core)

add_executable(tensor_05 test/tensor_05.cpp)
target_link_libraries(tensor_05 PRIVATE klygo_core)

add_executable(tensor_06 test/tensor_06.cpp)
target_link_libraries(tensor_06 PRIVATE klygo_core)

add_executable(tensor_07 test/tensor_07.cpp)
target_link_libraries(tensor_07 PRIVATE klygo_core)

add_executable(tensor_08 test/tensor_08.cpp)
target_link_libraries(tensor_08 PRIVATE klygo_core)