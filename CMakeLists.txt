cmake_minimum_required(VERSION 3.25.2)

set(CMAKE_BUILD_TYPE_INIT Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(SUBPROJECT_NAME feature-matcher)
project(${SUBPROJECT_NAME} LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include(GNUInstallDirs)

# OpenCV
find_package(OpenCV REQUIRED)

# CUDA
option(FEATURE_MATCHER_ENABLE_CUDA "Compile the CUDA device" OFF)
set(cuda ${FEATURE_MATCHER_ENABLE_CUDA})
if (cuda)
  enable_language(CUDA)
  find_package(CUDAToolkit)
endif()

#--------------------------------------------------------------------------------------------------

add_library(${SUBPROJECT_NAME} SHARED)
set_property(TARGET ${SUBPROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
if (cuda)
  add_library(${SUBPROJECT_NAME}_cuda SHARED)
endif()

set(SOURCES
    feature_matcher.cpp
    match_result.cpp
)

set(CUDA_SOURCES
    feature_matcher.cu
)

target_sources(${SUBPROJECT_NAME} PRIVATE ${SOURCES})

target_link_libraries(${SUBPROJECT_NAME} ${OpenCV_LIBS})

target_include_directories(${SUBPROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${OpenCV_INCLUDE_DIRS}
)

if (cuda)
  set_target_properties(${SUBPROJECT_NAME}_cuda PROPERTIES CUDA_ARCHITECTURES "native")

  target_sources(${SUBPROJECT_NAME}_cuda PRIVATE ${SOURCES})
  target_sources(${SUBPROJECT_NAME}_cuda PRIVATE ${CUDA_SOURCES})

  set_source_files_properties(${CUDA_SOURCES}
    PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
  )
  set_property(TARGET ${SUBPROJECT_NAME}_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(${SUBPROJECT_NAME}_cuda PRIVATE WITH_CUDA=1)

  target_link_libraries(${SUBPROJECT_NAME}_cuda ${OpenCV_LIBS})

  target_include_directories(${SUBPROJECT_NAME}_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${OpenCV_INCLUDE_DIRS}
  )
endif()