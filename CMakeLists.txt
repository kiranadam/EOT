cmake_minimum_required(VERSION 2.8) 
project(EOT)
find_package(OpenCV REQUIRED)
#find_package(Eigen3 REQUIRED)
set(NAME_SRC
    	src/test.cpp	
)

#set(NAME_HEADERS       
#        include/kalman.hpp
#)

# add a compiler flag for the C++2011 standard
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x -fopenmp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) # ${EIGEN3_INCLUDE_DIR})
link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(EOT ${NAME_SRC}) #  ${NAME_HEADERS})

target_link_libraries(EOT ${OpenCV_LIBS} ${Boost_LIBRARIES})
