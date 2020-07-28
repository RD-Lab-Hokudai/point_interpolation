cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)

project(TestVisualizer)

# Find installed Open3D, this imports
#   Open3D_INCLUDE_DIRS
#   Open3D_LIBRARIES
#   Open3D_LIBRARY_DIRS
#   Open3D_C_FLAGS
#   Open3D_CXX_FLAGS
#   Open3D_EXE_LINKER_FLAGS
if(WIN32)
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/CMake)
else()
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
    list(APPEND Open3D_LIBRARIES dl)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

# Set OS-specific things here
if(WIN32)
elseif(CYGWIN)
elseif(APPLE)
elseif(UNIX)
	add_definitions(-DUNIX)
	add_compile_options(-Wno-deprecated-declarations)
	add_compile_options(-Wno-unused-result)
    add_definitions(-O3)
endif(WIN32)

find_package( OpenCV REQUIRED )

# Open3D
if (Open3D_FOUND)
    message(STATUS "Found Open3D ${Open3D_VERSION}")

    # link_directories must be before add_executable
    link_directories(${Open3D_LIBRARY_DIRS})

    add_executable(TestVisualizer TestVisualizer.cpp)

    target_link_libraries(TestVisualizer ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(TestVisualizer PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(Segmentator Segmentator.cpp)

    target_link_libraries(Segmentator ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(Segmentator PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(ColorMRF ColorMRF.cpp)

    target_link_libraries(ColorMRF ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(ColorMRF PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(CG CG.cpp)

    target_link_libraries(CG ${Open3D_LIBRARIES})

    target_include_directories(CG PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(LU LU.cpp)

    target_link_libraries(LU ${Open3D_LIBRARIES})

    target_include_directories(LU PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(LM LM.cpp)

    target_link_libraries(LM ${Open3D_LIBRARIES})

    target_include_directories(LM PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(Linear Linear.cpp)

    target_link_libraries(Linear ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(Linear PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(Calibrator Calibrator.cpp)

    target_link_libraries(Calibrator ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(Calibrator PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(NeighborTest NeighborTest.cpp)

    target_link_libraries(NeighborTest ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(NeighborTest PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(PWAS_ICP PWAS_ICP.cpp)

    target_link_libraries(PWAS_ICP ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(PWAS_ICP PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(PWAS_RAW PWAS_RAW.cpp)

    target_link_libraries(PWAS_RAW ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(PWAS_RAW PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(CalibratorRGB CalibratorRGB.cpp)

    target_link_libraries(CalibratorRGB ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(CalibratorRGB PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(AutoCalibrator AutoCalibrator.cpp)

    target_link_libraries(AutoCalibrator ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(AutoCalibrator PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(AutoCalibrator2 AutoCalibrator2.cpp)

    target_link_libraries(AutoCalibrator2 ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(AutoCalibrator2 PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(CalibratorWithICP CalibratorWithICP.cpp)

    target_link_libraries(CalibratorWithICP ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(CalibratorWithICP PUBLIC ${Open3D_INCLUDE_DIRS})

    add_executable(PWAS PWAS.cpp)

    target_link_libraries(PWAS ${Open3D_LIBRARIES} ${OpenCV_LIBS})

    target_include_directories(PWAS PUBLIC ${Open3D_INCLUDE_DIRS})

    # Hot fix windows dll not found issue, assumming we're using the Release build
    option(BUILD_SHARED_LIBS "Whether Open3D was build as shared library" OFF)
    if(WIN32 AND BUILD_SHARED_LIBS)
        message("Will copy Open3D.dll to ${CMAKE_CURRENT_BINARY_DIR}/Release")
        add_custom_command(TARGET TestVisualizer POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy
                                ${CMAKE_INSTALL_PREFIX}/bin/Open3D.dll
                                ${CMAKE_CURRENT_BINARY_DIR}/Release)
    endif()

else ()
    message(SEND_ERROR "Open3D not found")
endif ()