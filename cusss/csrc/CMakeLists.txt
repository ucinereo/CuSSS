cmake_minimum_required(VERSION 3.18) # Bumped to 3.18 for better CUDA support
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
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH_OUTPUT}")
find_package(Torch REQUIRED)

add_library(cusss SHARED
    # SSS Kernel
    "ops/sss.cpp"
    "cuda/sss_cuda.cu"
    # SSSLU Kernel
    "ops/ssslu.cpp"
    "cuda/ssslu_cuda.cu"
    # xSSS Kernel
    "ops/xsss.cpp"
    "cuda/xsss_cuda.cu"
    # SSSGLU Kernel
    "ops/sssglu.cpp"
    "cuda/sssglu_cuda.cu"
    # xSSSLU Kernel
    "ops/xssslu.cpp"
    "cuda/xssslu_cuda.cu"


    # @TODO: Add other SSS variants here
)

# =========================================================================
# Optimization Flags

target_compile_options(cusss PRIVATE
    # C++ Host Flags
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
        -Wno-stringop-overflow  # Silence the GCC 13 std::vector warnings
        -Wno-maybe-uninitialized
    >

    # CUDA Device Flags
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math           # The most important speed flag
        --expt-relaxed-constexpr
        -Xptxas=-O3
    >
)

target_include_directories(cusss PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${TORCH_INCLUDE_DIRS}
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
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
)
