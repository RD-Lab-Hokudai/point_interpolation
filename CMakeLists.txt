cmake_minimum_required(VERSION 3.8.0)
set (CMAKE_CXX_STANDARD 17)

project(TestVisualizer)

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

add_executable(TestVisualizer TestVisualizer.cpp)

add_executable(CG CG.cpp)

add_executable(LU LU.cpp)

add_executable(LM LM.cpp)

add_executable(Calibrator Calibrator.cpp)

add_executable(CalibratorRGB CalibratorRGB.cpp)

add_executable(CalibratorZED CalibratorZED.cpp)

add_executable(AutoCalibrator AutoCalibrator.cpp)

add_executable(AutoCalibrator2 AutoCalibrator2.cpp)

add_executable(CalibratorWithICP CalibratorWithICP.cpp)

add_executable(Interpolater Interpolater.cpp)

add_executable(Tuner Tuner.cpp)

add_executable(InterpolateKitti InterpolateKitti.cpp)
