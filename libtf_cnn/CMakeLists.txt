cmake_minimum_required(VERSION 3.0.0)

project(tf_cnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")

set(TENSORFLOW_SOURCE_DIR /repos/tensorflow/tensorflow)

find_package(OpenCV REQUIRED)

aux_source_directory(. SRC_LIST)

include_directories (
	    ${TENSORFLOW_SOURCE_DIR}
 	    ${TENSORFLOW_SOURCE_DIR}/bazel-tensorflow/external/eigen_archive
	    ${TENSORFLOW_SOURCE_DIR}/bazel-tensorflow/external/com_google_absl
	    ${TENSORFLOW_SOURCE_DIR}/tensorflow/cc
	    ${TENSORFLOW_SOURCE_DIR}/tensorflow/core
	    ${TENSORFLOW_SOURCE_DIR}/tensorflow/contrib/makefile/gen/proto
	    ${TENSORFLOW_SOURCE_DIR}/tensorflow/contrib/makefile/gen/protobuf/include
	    ${OpenCV_INCLUDE_DIRS}
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/) 

add_library(tf_cnn SHARED ${SRC_LIST})
