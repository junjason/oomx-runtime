cmake_minimum_required(VERSION 3.22)
project(oomx LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optional: turn on later when LLVM/MLIR is ready
option(OOMX_WITH_LLVM "Build LLVM/MLIR dialects and pass" OFF)

# If/when you enable LLVM/MLIR, uncomment this block and set LLVM_DIR/MLIR_DIR or CMAKE_PREFIX_PATH
# if (OOMX_WITH_LLVM)
#   list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
#   find_package(LLVM 17 REQUIRED CONFIG)
#   find_package(MLIR REQUIRED CONFIG)
#   message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
#   include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
#   add_definitions(${LLVM_DEFINITIONS})
# endif()

add_subdirectory(runtime)
# if (OOMX_WITH_LLVM AND EXISTS "${CMAKE_SOURCE_DIR}/mlir/CMakeLists.txt")
#   add_subdirectory(mlir)
# endif()
# if (OOMX_WITH_LLVM AND EXISTS "${CMAKE_SOURCE_DIR}/passes/CMakeLists.txt")
#   add_subdirectory(passes)
# endif()
add_subdirectory(tools/oomx-bench)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
enable_language(CUDA)
add_subdirectory(gpu)
endif()