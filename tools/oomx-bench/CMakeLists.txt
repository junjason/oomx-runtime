add_executable(oomx-bench
  main.cpp
)
target_include_directories(oomx-bench PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(oomx-bench PRIVATE
  oomx_runtime
)
set_target_properties(oomx-bench PROPERTIES CXX_STANDARD 20)
