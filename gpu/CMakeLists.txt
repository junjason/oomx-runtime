add_library(oomx_cuda STATIC oomx_cuda.cu)
set_target_properties(oomx_cuda PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
POSITION_INDEPENDENT_CODE ON
)
target_include_directories(oomx_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
# Use a baseline arch; override with -DCMAKE_CUDA_ARCHITECTURES=86,90, etc.
if (NOT CMAKE_CUDA_ARCHITECTURES)
set_target_properties(oomx_cuda PROPERTIES CUDA_ARCHITECTURES "75;86")
endif()