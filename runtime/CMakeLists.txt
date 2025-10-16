add_library(oomx_runtime STATIC
  Runtime.cpp
  Store.cpp
  Sow.cpp
  Schema.cpp
)
target_include_directories(oomx_runtime PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(oomx_runtime PROPERTIES CXX_STANDARD 20)
