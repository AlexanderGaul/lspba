cmake_minimum_required(VERSION 3.5)

project(optimization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${CMAKE_SOURCE_DIR}/include/")

find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIRS})
message(STATUS "Eigen inlcude directories: ${EIGEN3_INCLUDE_DIRS}")

find_package(Ceres 2.0.0 REQUIRED)

find_package(Sophus REQUIRED)
message( STATUS "Sophus include directories: ${Sophus_INCLUDE_DIRS}" )
message( STATUS "Sophus library directories: ${Sophus_LIBRARY_DIRS}" )
message(STATUS "${Sophus_LIBRARIES}")
include_directories(${Sophus_INCLUDE_DIRS})

find_package(OpenCV 4.2.0 REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}" )
message(STATUS "OpenCV libraries ${OpenCV_LIBRARIES}")

#find_package(Open3D 0.12.0 REQUIRED)
#include_directories(SYSTEM ${Open3D_INCLUDE_DIRS})
#message(STATUS "Open3D include directories: ${Open3D_INCLUDE_DIRS}" )
#message(STATUS "Open3D libraries ${Open3D_LIBRARIES}")
#message(STATUS "Open3D library directories: ${Open3D_LIBRARY_DIRS}" )

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message( STATUS "OpenMP found" )
endif()
include_directories("/home/alexander/apps/basalt-headers/include/")

find_package(Python3 COMPONENTS Development NumPy REQUIRED)
find_package(PythonInterp 3.8 REQUIRED)
#find_package(PyhonLibs 3.8 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(PUBLIC_LIBRARIES Eigen3::Eigen Ceres::ceres  ${Sophus_LIBRARIES}  ${OpenCV_LIBRARIES})

#add_executable(robust_mean robust_mean.cpp)
#target_link_libraries(robust_mean ${CERES_LIBRARIES} ${Sophus_LIBRARIES} ${EIGEN_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(levenberg_marquardt lm.cpp)
target_link_libraries(levenberg_marquardt ${PUBLIC_LIBRARIES})
target_link_libraries(levenberg_marquardt Python3::Python Python3::NumPy)
target_include_directories(levenberg_marquardt PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${PYTHON_LIBRARIES})




