cmake_minimum_required(VERSION 3.8.0)
set (CMAKE_CXX_STANDARD 17)

project(PointInterpolation)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

add_definitions(-O3)

add_library(models include/models.h src/models.cpp)
add_library(methods include/utils.h src/utils.cpp include/methods.h src/methods.cpp)
add_library(preprocess include/preprocess.h src/preprocess.cpp)
add_library(postprocess include/postprocess.h src/postprocess.cpp)

link_directories(include ${PCL_LIBRARY_DIRS})
include_directories(PUBLIC include ${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARIES} ${OpenCV_LIBS} models methods preprocess postprocess)

add_executable(Interpolater src/Interpolater.cpp)

add_executable(Tuner src/Tuner.cpp)