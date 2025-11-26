cmake_minimum_required(VERSION 3.3)
project(cusss LANGUAGES CXX)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install DLL/binaries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

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

add_library(cusss SHARED
    # SSS Kernel
    "cpp/include/sss.hpp"
    "cpp/src/sss.cpp"
    "cuda/include/sss_impl.hpp"
    "cuda/src/sss_impl.cu"

    # @TODO: Add other SSS variants here
)

target_include_directories(cusss PRIVATE
    ${CMAKE_SOURCE_DIR}/cuda/include
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cusss PRIVATE 
    "${TORCH_LIBRARIES}" 
    CUDA::cudart
)

# Link cudadevrt if available (needed for device code)
if(TARGET CUDA::cudadevrt)
    target_link_libraries(cusss PRIVATE CUDA::cudadevrt)
endif()

target_compile_features(cusss PRIVATE cxx_std_17)

install(TARGETS cusss
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${BIN_INSTALL_DIR}
)
