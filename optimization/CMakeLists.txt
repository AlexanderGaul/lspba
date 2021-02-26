cmake_minimum_required(VERSION 3.5)

project(optimization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)


#set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -ggdb -g3 -DEIGEN_INITIALIZE_MATRICES_BY_NAN")      
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


find_package(Ceres 2.0.0 REQUIRED)

find_package(Sophus REQUIRED PATHS "/home/alexander/Documents/studies/20_ws/idp/code/libraries/Sophus/sophus/build")
message( STATUS "Sophus include directories: ${Sophus_INCLUDE_DIRS}" )
message( STATUS "Sophus library directories: ${Sophus_LIBRARY_DIR}" )
message(STATUS "${Sophus_LIBRARIES}")
include_directories("/home/alexander/Documents/studies/20_ws/idp/code/libraries/Sophus/")


include_directories("${CMAKE_SOURCE_DIR}/include")
message( STATUS "Include directories: ${CMAKE_SOURCE_DIR}/include" )

find_package(OpenCV 4.3.0 REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
#link_directories(${OpenCV_LIBRARY_DIRS})
message( STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}" )

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBRARIES}")

set(EIGEN3_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.3.7 REQUIRED)
set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Open3D 0.12.0 REQUIRED)
include_directories(SYSTEM ${Open3D_INCLUDE_DIRS})
link_directories(${Open3D_LIBRARY_DIRS})
message( STATUS "Open3D include directories: ${Open3D_INCLUDE_DIRS}" )
message(STATUS "Open3D libraries ${Open3D_LIBRARIES}")
message( STATUS "Open3D library directories: ${Open3D_LIBRARY_DIRS}" )

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message( STATUS "OpenMP found" )
endif()



set(PRIVATE_LIBRARIES Open3D::Open3D)
set(PUBLIC_LIBRARIES Ceres::ceres  ${Sophus_LIBRARIES}  ${Eigen_LIBRARIES}  ${OpenCV_LIBRARIES})


add_executable(robust_mean robust_mean.cpp)
target_link_libraries(robust_mean ${CERES_LIBRARIES} ${Sophus_LIBRARIES} ${EIGEN_LIBRARIES} ${OpenCV_LIBRARIES})

#
add_executable(levenberg_marquardt lm.cpp)
target_link_libraries(levenberg_marquardt  PRIVATE ${PUBLIC_LIBRARIES} PRIVATE ${PRIVATE_LIBRARIES} )
target_link_directories(levenberg_marquardt PRIVATE ${Open3D_LIBRARY_DIRS} PRIVATE ${OpenCV_LIBRARY_DIRS})

# PRIVATE "/usr/local/lib/"