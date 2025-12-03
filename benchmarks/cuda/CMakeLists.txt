cmake_minimum_required(VERSION 3.3)
project(cusss LANGUAGES CXX)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install DLL/binaries")

find_package(Python COMPONENTS Interpreter Development REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE)
else()
    message(FATAL_ERROR "Could not find CUDA Compiler. Make sure $CUDA_HOME is set.")
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT
    OUTPUT_VARIABLE TORCH_CMAKE_PATH_OUTPUT
    ERROR_VARIABLE TORCH_CMAKE_PATH_ERROR
)

if (NOT ${TORCH_CMAKE_PATH_RESULT} EQUAL 0)
    message(FATAL_ERROR "failed to find your pytorch installation, error: ${TORCH_CMAKE_PATH_ERROR}\n")
endif()

string(STRIP ${TORCH_CMAKE_PATH_OUTPUT} TORCH_CMAKE_PATH_OUTPUT)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PATH_OUTPUT}")

message(STATUS "TORCH_CMAKE_PATH_OUTPUT: ${TORCH_CMAKE_PATH_OUTPUT}")

find_package(Torch 2.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------
# Add nlohmann::json via FetchContent   <-- NEW
# ------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/latest/download/json.tar.xz
)

FetchContent_MakeAvailable(nlohmann_json)
# ------------------------------------------------------------

add_executable(cuda_benchmark
    "../../cusss/cpp/include/sss.hpp"
    "../../cusss/cpp/src/sss.cpp"
    "../../cusss/cuda/include/sss_impl.hpp"
    "../../cusss/cuda/src/sss_impl.cu"
    "src/kernels/all_kernels_import.hpp"
    "src/kernels/baseline_kernels/baseline_kernels.cu"
    "src/kernels/baseline_kernels/baseline_kernels.hpp"

    # Add the source containing main()
    "src/kernel_benchmark.cu"
)

target_include_directories(cuda_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cusss/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cusss/cuda/include
)

target_link_libraries(cuda_benchmark PRIVATE 
    "${TORCH_LIBRARIES}" 
    CUDA::cudart
    nlohmann_json::nlohmann_json              # <-- NEW
)

if(TARGET CUDA::cudadevrt)
    target_link_libraries(cuda_benchmark PRIVATE CUDA::cudadevrt)
endif()

target_compile_features(cuda_benchmark PRIVATE cxx_std_17)

install(TARGETS cuda_benchmark
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${BIN_INSTALL_DIR}
)
