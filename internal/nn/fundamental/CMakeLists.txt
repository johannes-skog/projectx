
project(tensor)

include(../cmake/Macros.cmake)

cmake_minimum_required(VERSION 2.8.9)

set(TARGET_NAME tensorl)

set(LINKER_LIBS "")

option(MESSAGE "" ON)
option(CUDA "Build with cuda support" ON)
option(OPENCV "Build with OpenCV support" ON)

if (CUDA)
  INCLUDE_CUDA()
endif(CUDA)

if (OPENCV)
  INCLUDE_OPENCV()
endif(OPENCV)

file(GLOB_RECURSE SOURCES_COMMON  "src/*.cpp" "src/*.cu" "include/*.cuh")

message("All source files" ${SOURCES_COMMON})

if (CUDA)

  #cuda_add_executable(main ${SOURCES_COMMON})

  cuda_add_library(${TARGET_NAME} STATIC ${SOURCES_COMMON})

  target_include_directories(${TARGET_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})

  target_include_directories(${TARGET_NAME} PUBLIC "${OpenCV_INCLUDE_DIRS}")

  target_include_directories(${TARGET_NAME} PUBLIC  include src)

  install(TARGETS ${TARGET_NAME} ARCHIVE DESTINATION lib/)

else ()

endif()

target_link_libraries(${TARGET_NAME} ${LINKER_LIBS})
