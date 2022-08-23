add_subdirectory(cuda)
add_subdirectory(frontend)
set(CMAKE_CUDA_ARCHITECTURES 75 80)
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")
