# CMake to build libpahof and test executable

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(cupahof)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
set (REQUIRED_LIBRARIES ${OpenCV_LIBS} )

INCLUDE(FindCUDA)

#------------------------------------------------
# DIRS
#------------------------------------------------
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(tests)

#-----------------------------------------------

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_10,code=sm_10)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

# About documentation
OPTION(INSTALL_DOC 	"Set to ON to build/install Documentation" OFF)
IF (INSTALL_DOC)
    FIND_PACKAGE(Doxygen REQUIRED)
    MESSAGE( STATUS "INSTALL_DOC:         ${INSTALL_DOC} ")
    INCLUDE("${PROJECT_SOURCE_DIR}/generateDoc.cmake")
    GENERATE_DOCUMENTATION(${PROJECT_SOURCE_DIR}/dox.in)
ENDIF()
