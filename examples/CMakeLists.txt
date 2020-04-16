project(tensor)

include(../cmake/Macros.cmake)

cmake_minimum_required(VERSION 3.3.2)

if (CUDA)
  INCLUDE_CUDA()
endif(CUDA)

set (CMAKE_CXX_STANDARD 14)

set( CMAKE_CXX_FLAGS "-O3 " )

option(MESSAGE "" ON)
option(CUDA "Build with cuda support" ON)
option(OPENCV "Build with OpenCV support" OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3")

file(GLOB_RECURSE SOURCES_COMMON  "../internal/src/*.cpp"
                                  "../internal/*.cu"
                                  "../internal/include/*.cuh")

function(BUILD_EXAMPLE TARGET MAIN_FILE)

  set(LINKER_LIBS)

  if (CUDA)

    file(GLOB files "src/*.cpp") # Compile all test with NVCC
    foreach(file ${files})
      message(${file})
      set_source_files_properties(${file}
                    PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
    endforeach()

    cuda_add_executable(${TARGET} ${MAIN_FILE} ${SOURCES_COMMON})

    INCLUDE_CUDA_DIRS(${TARGET})

    INCLUDE_CUBLAS(${TARGET})

  else()
    add_executable(${TARGET} ${MAIN_FILE} ${SOURCES_COMMON})
  endif()

  target_include_directories(${TARGET} PUBLIC ../external/catch2/)

  target_include_directories(${TARGET} PUBLIC ../external/stb/)

  target_include_directories(${TARGET} PUBLIC ../internal/nn/)

  target_link_libraries(${TARGET} ${LINKER_LIBS})

endfunction()

BUILD_EXAMPLE(example1 src/ex.cpp)

BUILD_EXAMPLE(example2 src/ex2.cpp)
