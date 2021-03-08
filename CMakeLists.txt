cmake_minimum_required(VERSION 3.8.0)
set (CMAKE_CXX_STANDARD 17)

project(PointInterpolation)

find_package(Open3D REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

add_definitions(-DUNIX)
add_compile_options(-Wno-deprecated-declarations)
add_compile_options(-Wno-unused-result)
add_definitions(-O3)

link_directories(${Open3D_LIBRARY_DIRS})
link_libraries(${Open3D_LIBRARIES} ${OpenCV_LIBS} stdc++fs)
include_directories(PUBLIC ${Open3D_INCLUDE_DIRS})

add_executable(Interpolater Interpolater.cpp)

add_executable(Tuner Tuner.cpp)

add_executable(InterpolateKitti InterpolateKitti.cpp)

add_executable(DistortionCorrector DistortionCorrector.cpp)
