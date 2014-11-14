#ifndef CUPAHOF_PARAMETERS_H
#define CUPAHOF_PARAMETERS_H

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <list>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>

#include <opencv/cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

#define JPG ".jpg"

//Images size
#define ROWS 128
#define COLS 128

const float _1_255 = 1./255;


//VECTOR_N = n_ang_*n_mags_ (8*4)
//This value must be changed if you use different parameters.
const int VECTOR_N = 32;

//Dot product kernel parameters.
const int ELEMENT_N =  ROWS*COLS;
const int SIZE_ROW = COLS;


#endif //CUPAHOF_PARAMETERS_H