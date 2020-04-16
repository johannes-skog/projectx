project(tensor)

include(../cmake/Macros.cmake)

cmake_minimum_required(VERSION 3.3.2)

set(TARGET_NAME tests)

set (CMAKE_CXX_STANDARD 14)

set( CMAKE_CXX_FLAGS "-O3 " )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++14 -fopenmp
   -march=native -fopenmp -mfpmath=sse -fPIC
   -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0;
  ")

option(MESSAGE "" ON)
option(CUDA "Build with cuda support" ON)
option(OPENCV "Build with OpenCV support" OFF)
option(OPENBLAS "Build with cuda support" ON)

if (CUDA)

  INCLUDE_CUDA()
  add_definitions(-DTENSOR_USE_CUDA)
  add_definitions(-DTENSOR_USE_CUDNN)

endif(CUDA)

if (OPENCV)
  INCLUDE_OPENCV()
endif(OPENCV)

if (OPENBLAS)
  INCLUDE_OPENBLAS(${TARGET_NAME})
endif(OPENBLAS)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3")

file(GLOB_RECURSE SOURCES_COMMON  "../internal/*.cpp"
                                  "modules/*.cpp"
                                  "modules/*.cu"
                                  "../internal/nn/*.cu"
                                  "../internal/include/*.cuh")


if (CUDA)

  file(GLOB files "modules/*.cpp") # Compile all test with NVCC
  foreach(file ${files})
    message(${file})
    set_source_files_properties(${file}
                  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
  endforeach()

  cuda_add_executable(${TARGET_NAME} main.cu ${SOURCES_COMMON})

  INCLUDE_CUDA_DIRS(${TARGET_NAME})


  INCLUDE_CUBLAS(${TARGET_NAME})

  INCLUDE_OPENBLAS_DIRS(${TARGET_NAME})

else()

  add_executable(${TARGET_NAME} main.cpp ${SOURCES_COMMON})

endif()


target_include_directories(${TARGET_NAME} PUBLIC ../external/catch2/)

target_include_directories(${TARGET_NAME} PUBLIC ../external/stb/)

#target_include_directories(${TARGET_NAME} PUBLIC ../internal/nn/fundamental)

target_include_directories(${TARGET_NAME} PUBLIC ../internal/nn/)

target_link_libraries(${TARGET_NAME} ${LINKER_LIBS})

add_test( NAME ${TARGET_NAME} COMMAND ${TARGET_NAME} -o report.xml -r junit)
