cmake_minimum_required(VERSION 3.25.2)

set(CMAKE_BUILD_TYPE_INIT Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(SUBPROJECT_NAME feature-matcher)
set(TARGET_ORB "${SUBPROJECT_NAME}-orb")
set(TARGET_SURF "${SUBPROJECT_NAME}-surf")
set(TARGET_CUDA "${SUBPROJECT_NAME}-cuda")
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

add_library(${TARGET_ORB} SHARED)
add_library(${TARGET_SURF} SHARED)
set_property(TARGET ${TARGET_ORB} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${TARGET_SURF} PROPERTY POSITION_INDEPENDENT_CODE ON)
if (cuda)
  add_library(${TARGET_CUDA} SHARED)
endif()

set(SOURCES
    feature_matcher.cpp
    match_result.cpp
)

set(SOURCES_ORB
    ${SOURCES}
    orb_matcher.cpp
)

set(SOURCES_SURF
    ${SOURCES}
    surf_matcher.cpp
)

set(CUDA_SOURCES
    ${SOURCES}
    feature_matcher.cu
    #cuda_matcher.cu
)

target_sources(${TARGET_ORB} PRIVATE ${SOURCES_ORB})
target_sources(${TARGET_SURF} PRIVATE ${SOURCES_SURF})

target_link_libraries(${TARGET_ORB} ${OpenCV_LIBS})
target_link_libraries(${TARGET_SURF} ${OpenCV_LIBS})

target_include_directories(${TARGET_ORB} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(${TARGET_SURF} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${OpenCV_INCLUDE_DIRS}
)

if (cuda)
  set_target_properties(${TARGET_CUDA} PROPERTIES CUDA_ARCHITECTURES "native")

  target_sources(${TARGET_CUDA} PRIVATE ${CUDA_SOURCES})

  set_source_files_properties(${CUDA_SOURCES}
    PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
  )
  set_property(TARGET ${TARGET_CUDA} PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(${TARGET_CUDA} PRIVATE WITH_CUDA=1)

  target_link_libraries(${TARGET_CUDA} ${OpenCV_LIBS})

  target_include_directories(${TARGET_CUDA} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${OpenCV_INCLUDE_DIRS}
  )
endif()