cmake_minimum_required(VERSION 2.8)
# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include_directories(${PROJECT_SOURCE_DIR})
project( maxflow_mincut )
find_package( OpenCV REQUIRED )
add_executable( maxflow main.cpp maxflow.cpp)
target_link_libraries( maxflow ${OpenCV_LIBS} )

