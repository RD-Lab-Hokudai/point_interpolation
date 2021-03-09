cmake_minimum_required(VERSION 3.8.0)
set (CMAKE_CXX_STANDARD 17)

project(PointInterpolation)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

link_directories(${PCL_LIBRARY_DIRS})
link_libraries(${PCL_LIBRARIES} ${OpenCV_LIBS})
include_directories(PUBLIC ${PCL_INCLUDE_DIRS})

add_executable(Interpolater Interpolater.cpp)

# add_executable(Tuner Tuner.cpp)

# add_executable(InterpolateKitti InterpolateKitti.cpp)

add_executable(DistortionCorrector DistortionCorrector.cpp)
