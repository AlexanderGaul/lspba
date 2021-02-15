cmake_minimum_required(VERSION 3.5)

project(optimization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#find_package(Eigen3 3.3.8 REQUIRED)

#list(APPEND GLOG_INCLUDE_DIR_HINTS "/usr/include")
#get_filename_component(CERES_BUILD_GLOG_LIBRARY_DIR "/usr/lib/libglog.so" PATH)
#list(APPEND GLOG_LIBRARY_DIR_HINTS ${CERES_BUILD_GLOG_LIBRARY_DIR})

# Search quietly s/t we control the timing of the error message if not found.
# PATHS "/usr/lib/cmake/Ceres/")

find_package(Ceres 2.0.0 REQUIRED)

find_package(Glog REQUIRED PATHS "/home/alexander/Documents/studies/20_ws/idp/code/libraries/glog/")
set(GLOG_INCLUDE_DIR ${GLOG_INCLUDE_DIR})
include_directories(${GLOG_INCLUDE_DIR})

#find_package(gflags REQUIRED)

find_package(Sophus REQUIRED PATHS "/home/alexander/Software/Sophus/build/")
message( STATUS "Sophus include directories: ${Sophus_INCLUDE_DIRS}" )
message( STATUS "Sophus library directories: ${Sophus_LIBRARY_DIR}" )
message(STATUS "${Sophus_LIBRARIES}")
include_directories("/home/alexander/Software/Sophus/")

#find_package(glog REQUIRED CONFIG NAMES google-glog glog)

include_directories("${CMAKE_SOURCE_DIR}/include")
message( STATUS "Include directories: ${CMAKE_SOURCE_DIR}/include" )

find_package(OpenCV 4.3.0 REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIR})
message( STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}" )
message( STATUS "OpenCV library directories: ${OpenCV_LIBRARY_DIR}" )

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBRARIES}")

set(EIGEN3_INCLUDE_DIR_HINTS ${EIGEN_ROOT})  # for opengv / pangolin
find_package(Eigen3 3.3.8 REQUIRED) # find eigen with our adapted "config module", that uses our eigen source tree
set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})  # for opengv / pangolin
include_directories(${EIGEN3_INCLUDE_DIR})

# Search quietly s/t we control the timing of the error message if not found.
#find_package(Glog REQUIRED PATHS "/home/alexander/Software/ceres-solver/src/ceres-solver-2.0.0/cmake/")

add_executable(robust_mean main.cpp)
target_link_libraries(robust_mean ${CERES_LIBRARIES} glog::glog ${Sophus_LIBRARIES} ${EIGEN_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(levenberg_marquardt lm.cpp)
target_link_libraries(levenberg_marquardt ${CERES_LIBRARIES} glog::glog ${Sophus_LIBRARIES} ${EIGEN_LIBRARIES} ${OpenCV_LIBRARIES})
