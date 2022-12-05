cmake_minimum_required(VERSION 3.23)
project(PhotometricCalibration)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS
        "${SSE_FLAGS} -std=c++0x "#${CMAKE_CXX_FLAGS_RELEASE} "
        )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(camera_response response.cpp)
target_link_libraries(camera_response ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(vignette_calibration vignette.cpp)
target_link_libraries(vignette_calibration ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(ArMarker ArMarker.cpp)
target_link_libraries(ArMarker ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(ApplyCalibration apply_calibration.cpp)
target_link_libraries(ApplyCalibration ${OpenCV_LIBS})