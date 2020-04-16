macro(INCLUDE_CUDA )

  find_package(CUDA 10.0 REQUIRED)

  if (MESSAGE)
    message("CUDA INCLUDE DIR " ${CUDA_INCLUDE_DIRS})
    message("CUDA LIBS " ${CUDA_LIBRARIES})
  endif()

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -gencode arch=compute_61,code=sm_61;
        )
  # Make sure that you are using the correct arch, gencode

  # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

  add_definitions(-DTENSOR_USE_CUDA)

  add_definitions(-DTENSOR_USE_CUDNN)

endmacro()

macro(INCLUDE_CUDA_DIRS TARGET)

  target_include_directories(${TARGET} PUBLIC ${CUDA_INCLUDE_DIRS})

endmacro()

macro(INCLUDE_CUBLAS TARGET)

  add_definitions(-DTENSOR_USE_CUBLAS)
  # FIX, make generic
  list(APPEND LINKER_LIBS  /usr/lib/x86_64-linux-gnu/libcublas.so)
  list(APPEND LINKER_LIBS  /usr/local/cuda/lib64/libcudnn.so)

  target_include_directories(${TARGET} PUBLIC /usr/include/)

endmacro()

macro(INCLUDE_OPENCV TARGET)

  set(CMAKE_PREFIX_PATH "/usr/local/lib/cmake/opencv4")
  find_package(OpenCV REQUIRED)

  if (MESSAGE)
    message("OPENCV INCLUDE DIR" ${OpenCV_INCLUDE_DIRS})
    message("OPENCV LIBS" ${OpenCV_LIBRARIES} )
  endif()

  list(APPEND LINKER_LIBS ${CUDA_LIBRARIES})

  add_definitions(-DTENSOR_USE_OPENCV)

endmacro()

macro(INCLUDE_OPENBLAS TARGET)

  list(APPEND LINKER_LIBS /opt/OpenBLAS/lib/libopenblas.so)

  add_definitions(-DTENSOR_USE_OPENBLAS)

endmacro()

macro(INCLUDE_OPENBLAS_DIRS TARGET)

  target_include_directories(${TARGET} PUBLIC /opt/OpenBLAS/include/)

endmacro()

macro(INCLUDE_OPENCV_DIRS TARGET)

  target_include_directories(${TARGET} PUBLIC "${OpenCV_INCLUDE_DIRS}")

endmacro()
