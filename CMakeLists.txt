cmake_minimum_required(VERSION 3.0.0)

project(tfserver)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

link_directories(./libtf_cnn)

include_directories(
	    ${PROJECT_SOURCE_DIR}/libtf_cnn
)

set(LIBS -ljsoncpp -lmicrohttpd -ljsonrpccpp-common -ljsonrpccpp-server)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}) 

add_executable(tfserver main.cc)

target_link_libraries(tfserver ${OpenCV_LIBS} ${LIBS} tf_cnn tensorflow_cc)
