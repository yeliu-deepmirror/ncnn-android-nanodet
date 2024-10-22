project(nanodetncnn)

cmake_minimum_required(VERSION 3.22.1)

find_library(log-lib log)
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.10.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240820-android-vulkan-shared/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/nanodet)
include_directories(${CMAKE_SOURCE_DIR}/common)

add_library(nanodetncnn SHARED
  nanodet/nanodetncnn.cpp
  nanodet/nanodet.cpp
  ndkcamera.cpp
)

target_link_libraries(nanodetncnn
  ncnn
  ${OpenCV_LIBS}
  camera2ndk
  mediandk
  -landroid # Add this.
  ${log-lib}
)
