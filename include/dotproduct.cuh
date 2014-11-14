/**
* \file dotproduct.cuh
*  This header file contains the definition of the method that
*  will call the main kernel of cupahof
*
* @author Fernando Cobo Aguilera
* @date October 2014
*/


#ifndef CUPAHOF_DOTPRODUCT_H
#define CUPAHOF_DOTPRODUCT_H


#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include "parameters.h"


/**
* This function calls the kernel DotProductKernel where a dot product is executed in the gpu
* \param d_matrix_a First gpu memory matrix to be multiplied
* \param d_matrix_b Second gpu memory matrix to be multiplied
* \param stream Stream for the asynchronous version
* \param d_dot_result Gpu memory matrix where the dot product will be saved
*/
void DotProduct(cv::gpu::GpuMat d_matrix_a,cv::gpu::GpuMat d_matrix_b, cudaStream_t stream,cv::gpu::GpuMat d_dot_result);


#endif //CUPAHOF_DOTPRODUCT_H