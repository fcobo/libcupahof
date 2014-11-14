/**
* \file paHof.h
* This file contains the definitions of  the main classes of the algorithm where
* PAHOFCUDA is calculated
*
* @author Fernando Cobo Aguilera
* @date October 2014
*/

#ifndef CUPAHOF_PAHOF_H
#define CUPAHOF_PAHOF_H


#include "parameters.h"
#include "utilities.h"
#include "dotproduct.cuh"

/** 
* \struct BufferHof
* This struct helps to manage gpu memory while Hof is being calculated

* Memory allocation on the GPU is considerable. Therefore, if it’s possible allocate new memory 
* as few times as possible.
*/
struct BufferHof{

	cv::gpu::GpuMat d_histogram; //Gpu memory matrix with a histogram
	cv::Mat h_histogram; //Cpu memory matrix with a histogram
	cudaStream_t *streams; //Streams for the asynchronous version

	//Memory which is needed to calculate previous calculations to the histogram
	cv::gpu::CudaMem *h_sum_val_ang;
	cv::gpu::GpuMat *d_sum_val_ang;
	cv::Mat poshisto;
	cv::Mat poshi;
};


/**
* \class Hof
*
* \brief This is a class to calculate the histograms of PAHOFCUDA 
*
* This class provides the user a set of useful methods 
* to calculate histograms of PAHOFCUDA. It offers either a CPU version
- or a GPU version
*
* \author Fernando Cobo Aguilera
* \date 17/10/2014
*/
class Hof
{
public:

	/**
	* Class constructor
	*/
	Hof();

	/**
	* Class constructor
	* \param rows Number of rows of images
	* \param cols Number of cols of images
	* \param frames Number of frames that will be processed
	* \param ang Number of bins for optical flow angle quantization
	* \param mag Magnitude intervals for quantization
	* \param size_mag size of mag
	*/
	Hof(int rows, int cols, int frames, int ang,float *mag, int size_mag);

	/**
	* Class constructor
	* \param rows Number of rows of images
	* \param cols Number of cols of images
	* \param frames Number of frames that will be processed
	*/
	Hof(int rows, int cols, int frames);

	/**
	* Class destructor
	*/
	~Hof(void);

	/**
	* This method sets frames to n
	* \param n_frames Number of frames
	*/
	void SetFrames(int n_frames);

	/**
	* This method calculates the conversion from Cartesian to Polar. In this case,
	* pinned memory is used, that is to say, this calculation is executed in the Gpu and after that,
	* the result is downloaded to pinned memory
	* \param h_flow_u Cpu memory matrix with the component x of the optical flow
	* \param h_flow_v  Cpu memory matrix with the component y of the optical flow
	* \param h_magnitude Cpu memory matrix where coordinates magnitude will be saved
	* \param h_angle Cpu memory matrix where coordinates angle will be saved
	*/
	void CalculateCartToPolarGpuPinnedMemory(const cv::Mat h_flow_u,const cv::Mat h_flow_v, cv::gpu::CudaMem &h_magnitude, cv::gpu::CudaMem &h_angle);
	
	/**
	* This method calculates the conversion from Cartesian to Polar. 
	* This calculation is executed in the cpu, thus, the results are
	* downloaded to pageable memory
	* \param h_flow_u Cpu memory matrix with the component x of the optical flow
	* \param h_flow_v Cpu memory matrix with the component y of the optical flow
	* \param h_magnitude Cpu memory matrix where coordinates magnitude will be saved
	* \param h_angle Cpu memory matrix where coordinates angle will be saved
	*/
	void CalculateCartToPolarCpuPageableMemory(const cv::Mat h_flow_u,const cv::Mat h_flow_v, cv::Mat &h_magnitude, cv::Mat &h_angle);

	//cv::Mat ComputePreHofCpu(int tidx, int tidy);
	//cv::gpu::GpuMat ComputePreHofGpu(int tidx, int tidy);

	/**
	* This method computes the HOF (histograms of optical flow) of the PAHOFCUDA algorithm in the cpu.
	* \param x_tiles Subdivisions to the component x of the optical flow
	* \param y_tiles Subdivisions to the component y of the optical flow
	* \param h_angles Cpu memory matrix where coordinates angle are saved
	* \param h_magnitudes Cpu memory matrix where coordinates magnitude are saved
	* \param n_histograms Total number of histograms that will be executed, depending on the number of frames
	* \return A cpu memory matrix where every single row contains a histogram
	*/
	cv::Mat ComputeHofCpu(int x_tiles, int y_tiles, cv::Mat h_angles, cv::Mat h_magnitudes, int n_histograms);

	/**
	* This method computes the HOF (histograms of optical flow) of the PAHOFCUDA algorithm in the gpu.
	* \param x_tiles Subdivisions to the component x of the optical flow
	* \param y_tiles Subdivisions to the component y of the optical flow
	* \param h_angles Cpu memory matrix where coordinates angle are saved
	* \param h_magnitudes Cpu memory matrix where coordinates magnitude are saved
	* \param n_histograms Total number of histograms that will be executed, depending on the number of frames
	* \return A cpu memory matrix where every single row contains a histogram
	*/
	cv::Mat ComputeHofGpu(int x_tiles, int y_tiles, cv::Mat h_angles, cv::Mat h_magnitudes, int n_histograms);

	/**
	* This method computes one level of the HOF pyramid (histograms of optical flow) but the first. This method
	* is an cpu optimization of ComputeHofCpu which is based on sums
	* \param hof Matrix with the histograms of the previous level of the pyramid
	* \param x_tiles Subdivisions to the component x of the optical flow
	* \param y_tiles Subdivisions to the component y of the optical flow
	* \return A matrix with the level of the HOF pyramid calculated
	*/
	cv::Mat PyramidalHof(cv::Mat hof, int x_tiles, int y_tiles);

	/**
	* Getter n_ang_
	* \return n_ang_
	*/
	int GetNAng();

	/**
	* Getter mags_
	* \return mags_
	*/
	float * GetMags();

	/**
	* Getter n_mags_
	* \return n_mags_
	*/
	int GetNMags();

	//!Object to measure times
	Times timer;

private:

	/**
	* This method defines the masks that will be needed to calculate the HOF 
	* \param x_tiles Subdivisions to the component x of the optical flow
	* \param y_tiles Subdivisions to the component y of the optical flow
	*/
	void CalculateMasks(int x_tiles, int y_tiles);

	/**
	* This method calculates the weights that will be needed to calculate the HOF.
	* It uses the masks from CalculateMasks()
	* \param x_tiles Subdivisions to the component x of the optical flow
	* \param y_tiles Subdivisions to the component y of the optical flow
	* \return Pageable memory matrix with the weights
	*/
	cv::Mat CalculateWeights(int x_tiles, int y_tiles);

	/**
	* This method calculates the weights that will be needed to calculate the HOF.
	* This method is useful when HOF is going to be calculated in the GPU.
	* \param x_tiles Subdivisions to the component x of the optical flow
	* \param y_tiles Subdivisions to the component y of the optical flow
	* \return Pinned memory matrix with the weights
	*/
	cv::gpu::CudaMem CalculateWeightsCudaMem(int x_tiles, int y_tiles);


	/**
	* This method does comparations between the optical flow magnitude and the different
	* values that have been established. Therefore, we determinate the interval to which every
	* element of the magnitude belongs. This method is calculated in the cpu with pageable memory
	* \param h_angle Cpu memory matrix where coordinates angle are saved
	* \param h_magnitude  Cpu memory matrix where coordinates magnitude are saved
	* \return It returns a matrix with the results
	*/
	cv::Mat CalculateHistogramComparisons(cv::Mat h_angle, cv::Mat h_magnitude);

	/**
	* This method does comparations between the optical flow magnitude and the different
	* values that have been established. Therefore, we determinate the interval to which every
	* element of the magnitude belongs. This method is calculated in the cpu with pinned memory
	* \param h_angle Cpu memory matrix where coordinates angle are saved
	* \param h_magnitude  Cpu memory matrix where coordinates magnitude are saved
	* \param index Index of the matrix h_sum_val_ang in buffer, that will be used
	* \param buffer Buffer with memory that is needed
	*/
	void CalculateHistogramComparisonsCudaMem(cv::Mat h_angle, cv::Mat h_magnitude, int index, BufferHof buffer);

	/**
	* This method executes, in a quad loop, the dot product which is needed to calculate HOF. This version is
	* executed in the cpu
	* \param x_tiles Subdivisions to the component x of the optical flow
	* \param y_tiles Subdivisions to the component y of the optical flow
	* \param h_sum_val_ang Cpu memory matrix with previous calculations (CalculatePreHistogramCpu)
	* \param h_weight Cpu memory matrix with the weights (CalculateWeights)
	* \param h_histogram Matrix where the final histograms will be saved
	*/
	void CalculateHistogramDotProductCpu(int x_tiles, int y_tiles, cv::Mat h_sum_val_ang, cv::Mat h_weight, cv::Mat h_histogram);

	/**
	* This method executes, in a quad loop, the dot product which is needed to calculate HOF. This version is
	* executed in the gpu
	* \param x_tiles Subdivisions to the component x of the optical flow
	* \param y_tiles Subdivisions to the component y of the optical flow
	* \param d_sum_val_ang Gpu memory matrix with previous calculations (CalculatePreHistogramCpu)
	* \param d_weight Gpu memory matrix with the weights (CalculateWeights)
	* \param buffer Buffer where the histograms will be saved
	* \param index Index of the streams array that will be used
	*/
	void CalculateHistogramDotProductGpu(int x_tiles, int y_tiles, cv::gpu::GpuMat d_sum_val_ang,cv::gpu::GpuMat d_weight, BufferHof buffer, int index);

	
	//cv::Mat CalculateHistogramCpuV1(cv::Mat U,cv::Mat V, int nAng, int XTiles, int YTiles, float *mags);
	//cv::Mat NormalizeHistogram(cv::Mat H, int nMags);
	//cv::Mat ComputeaHof(cv::Mat HOF,int len, int step, bool doNorm, int nMags);

	//!Masks
	cv::Mat mask_x_;
	cv::Mat mask_y_;
	vector<cv::Mat> mod_mask_;

	//!Rows of the images
	int rows_;

	//!Cols of the images
	int cols_;

	//!Total number of frames
	int frames_;

	//! Number of bins for Optical Flow angle quantization
	int n_ang_;

	//! Magnitude intervals for quantization
	float *mags_;

	//! Size of mags
	int n_mags_;
};



/**
* \class AHof
*
* \brief This is a class to generate the motion descriptors. It accumulates
* HOFs to generate them.
*
* \author Fernando Cobo Aguilera
* \date 17/10/2014
*/
class AHof
{
public:

	/**
	* Class constructor
	*/
	AHof(void);

	/**
	* Class constructor
	* \param leng  Number of frames to accumulate
	* \param st Every 'st' frames a descriptor is computed
	*/
	AHof(int leng, int st);

	/**
	* Class destructor
	*/
	~AHof(void);

	/**
	* This method computes the HOF accumulation
	* \param h_hof A matrix with the histograms (a histogram every row)
	* \return It returns a matrix with the HOF accumulation 
	*/
	cv::Mat ComputeAhof(cv::Mat h_hof);

	/**
	* This method computes the optimized HOF accumulation. In this case,
	* some operations are eliminated to avoid repetition
	* \param h_hof A matrix with the histograms (a histogram every row)
	* \return It returns a matrix with the HOF accumulation 
	*/
	cv::Mat ComputeAhofOptimized(cv::Mat h_hof);


	/**
	* This method normalizes the HOF accumulation
	* \param nMags Size of magnitude intervals for quantization
	*/
	cv::Mat NormalizeAhof(int nMags, cv::Mat ahof);
	
private:

	//! Number of frames to accumulate
	int len_;

	//! Every 'step_' frames a descriptor is computed
	int step_;
};

/**
* \class PAHof
*
* \brief This class executes the algorithm PaHOF in the CPU or the GPU
*
* \author Fernando Cobo Aguilera
* \date 17/10/2014
*/
class PAHof{

public:

	/**
	* Class constructor
	*/
	PAHof();

	/**
	* Class constructor
	* \param ang Number of bins for Optical Flow angle quantization
	* \param mag Magnitude intervals for quantization
	* \param leng Number of frames to accumulate
	* \param st Every 'st' frames a descriptor is computed
	* \param frames Total number of frames
	* \param rows Total number of rows of the images
	* \param cols Total number of cols of the images
	* \param xytiles Subdivisions to the components of the optical flow
	*/
	PAHof(int ang, float *mag, int leng, int st, int frames, int rows, int cols, cv::Mat xytiles, int size_mag);

	/**
	* Class destructor
	*/
	~PAHof(void);

	/**
	* This method calculates the optical flow of a set of images
	* \param base_name Common name of the files where the images are saved. 
	* The full name should be as follow: base_name + number + .jpg
	* \param h_flow_u Cpu memory matrix where the component x of the optical flow will be saved
	* \param h_flow_v Cpu memory matrix where the component y of the optical flow will be saved
	*/
	void CalculateOpticalFlowFromImages(string base_name,cv::Mat &h_flow_u, cv::Mat &h_flow_v);

	/**
	* This method loads the optical flow from xml files
	* \param base_name_of_u Common name of the files where the component x of the optical flow is saved. 
	* The full name should be as follow: base_name + U + number + .xml
	* \param base_name_of_v Common name of the files where the component y of the optical flow is saved. 
	* The full name should be as follow: base_name + V + number + .xml
	* \param h_flow_u Cpu memory matrix where the component x of the optical flow will be saved
	* \param h_flow_v Cpu memory matrix where the component y of the optical flow will be saved
	*/
	void LoadOpticalFlowFromFiles(string base_name_of_u,string base_name_of_v,cv::Mat &h_flow_u, cv::Mat &h_flow_v);

	/**
	* This method calculates the algorithm PAHOFCUDA using the gpu (fast version)
	* \param h_flow_u Cpu memory matrix with the component x of the optical flow 
	* \param h_flow_v Cpu memory matrix with the component y of the optical flow
	* \param acc_optimized If true, the accumulation will be optimized (insignificant difference)
	* \param pyramid_levels Total number of pyramid levels that will be executed without optimizing
	* (Only 1 is the best option)
	*/
	cv::Mat PAHofGpu(cv::Mat h_flow_u, cv::Mat h_flow_v, bool acc_optimized, int pyramid_levels = 1);

	/**
	* This method calculates the algorithm PAHOFCUDA using the cpu (slow version)
	* \param h_flow_u Cpu memory matrix with the component x of the optical flow 
	* \param h_flow_v Cpu memory matrix with the component y of the optical flow
	* \param acc_optimized If true, the accumulation will be optimized (insignificant difference)
	* \param pyramid_levels Total number of pyramid levels that will be executed without optimizing
	* (Only 1 is the best option)
	*/
	cv::Mat PAHofCpu(cv::Mat h_flow_u, cv::Mat h_flow_v, bool acc_optimized, int pyramid_levels = 1);

	/**
	* Setter of frames_
	* \param frame Number of frames
	*/
	void SetFrames(int frames);


private:

	//!Subdivisions to the components of the optical flow
	cv::Mat x_y_tiles_;

	//!Rows of the images
	int rows_;

	//!Cols of the images
	int cols_;

	//!Total number of frames
	int frames_;

	//!Optical flow manager
	OpticalFlowManagement optical_flow_management_;

	//!Hof object
	Hof histograms_optical_flow_;

	//!AHof object
	AHof accumulated_histograms_optical_flow_;

	//!Check sums
	int* sums_;
};


#endif //CUPAHOF_PAHOF_H