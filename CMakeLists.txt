cmake_minimum_required(VERSION 3.5)
project(SnpeLoaderTest)

set(CMAKE_CXX_STANDARD 17)

set(SNPE_ROOT $ENV{SNPE_ROOT})
if(NOT SNPE_ROOT)
    message(FATAL_ERROR "SNPE_ROOT not set!")
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")


include_directories(${SNPE_ROOT}/include/SNPE)
include_directories(${OpenCV_INCLUDE_DIRS})


link_directories(${SNPE_ROOT}/lib/aarch64-oe-linux-gcc11.2)

add_executable(load_test load_test.cpp)

target_link_libraries(load_test 
    SNPE            
    ${OpenCV_LIBS}  
    dl              
    pthread         
)