/**
* \file utilities.h
* This header file contains the declaration of classes that are needed
* for the PaHof demo. It also contains the implementation of an usefull class
* which shows algorithm times. Last but not least, a tester command line arguments 
* class has been added too.

* @author Fernando Cobo Aguilera
* @date October 2014
*/

#ifndef CUPAHOF_UTILITIES_H
#define CUPAHOF_UTILITIES_H

#include "parameters.h"

#if  defined _WIN32
#else
#include <sys/time.h>
#endif

/**
* \class Utilities
* \brief This class provides a group of useful methods to manipulate images
*
* \author Fernando Cobo Aguilera
* \date 30/09/2014
*/
class Utilities{

public:

	/**
	* This method resizes a set images and saves them in memory
	* \param number_of_frames Total number of images
	* \param file_name_in Common name of the files where the images are saved. The full name should be
	* as follow: file_name_in + number + .jpg
	* \param file_name_out Common name of the files where the new images will be saved. The full name should be
	* as follow: file_name_out + number + .jpg
	* \param size New size
	*/
	static void ResizeImages(int number_of_frames, string file_name_in, string file_name_out,cv::Size size);

	/**
	* This function saves optical flow to different files. This function is only
	* available when the optical flow is in vectors
	* \param h_flow_u An array of cpu memory matrices with every component x of the optical flow
	* \param h_flow_v An array of gpu memory matrices with every component x of the optical flow
	* \param base_name_u Common name of the files where the component x of theoptical flow will be saved. 
	* The full name should be as follow: base_name + number + .jpg
	* \param base_name_v Common name of the files where the component y of theoptical flow will be saved. 
	* The full name should be as follow: base_name + number + .jpg
	*/
	static void WriteFlowToFiles(vector<cv::Mat> h_flow_u, vector<cv::Mat> h_flow_v, string base_name_u, string base_name_v);

	/**
	* This method saves a Mat into a Mat-File of Matlab
	* \param mat Data that will be saved
	* \param file_name Name of the file
	* \param var_name of the variable in Matlab
	* \param bgr_to_rgb Transform from a bgr mat to a rgb one
	*/
	static void WriteMat(cv::Mat const& mat, const char* file_name, const char* var_name = "A", bool bgr_to_rgb = true);
};


/**
* \class OpticalFlowManagement
*
* \brief This is a class to calculate the Farneback optical flow in OpenCV with a gpu. 
*
* This class provides the user a complete collection of useful methods 
* to calculate the optical flow, to manipulate it and to load and download it
* from GPU to CPU or vice versa. Futhermore, it contains different versions
* depending on the type of GPU memory used.
*
* \author Fernando Cobo Aguilera
* \date 30/09/2014
*/

class OpticalFlowManagement
{
public:

	/**
	* Class constructor
	*/
	OpticalFlowManagement();

	/**
	* Class constructor
	* \param rows Number of rows
	* \param cols Number of cols
	* \param frames Total number of frames
	*/
	OpticalFlowManagement(int rows, int cols, int frames);

	/**
	* Class constructor
	* \param frames Total number of frames
	* \param file Common name of the images that will be uploaded
	*/
	OpticalFlowManagement(int frames);

	/**
	* Class destructor
	*/
	~OpticalFlowManagement(void);

	/** 
	* This function loads the images from files to pinned memory and, after that,
	* they are uploaded to gpu memory. Each image is saved in memory below the previos one,
	* avoiding possible access problems.
	* \param base_name Common name of the files where the images are saved. The full name should be
	* as follow: baseName + number + .jpg
	* \return A gpu memory matrix with the images already loaded
	*/
	cv::gpu::GpuMat LoadImagesFromCpuToGpuPinnedMemory(string base_name);

	/** 
	* This function loads the images from files to memory and, after that,
	* they are uploaded to gpu memory. Each image is saved in memory below the previos one,
	* avoiding possible access problems. This kind of upload is slower than the pinned one
	* \param base_name Common name of the files where images are saved. The full name should be
	* as follow: base_name + number + .jpg
	* \return A gpu memory matrix with the images already loaded
	*/
	cv::gpu::GpuMat LoadImagesFromCpuToGpuPageableMemory(string base_name);

	/**
	* This function calculates the optical flow of a group of frames which are already loaded in
	* memory. The function will save the results in two vectors (x and y component). Each vector
	* will contain all the optical flow mats depending on the total number of frames.
	* \param d_images A gpu memory matrix with all the images already loaded
	* \param d_flow_u An array of gpu memory matrices where every component x of the optical flow will be saved
	* \param d_flow_v An array of gpu memory matrices where every component y of the optical flow will be saved
	*/
	void CalculateOpticalFlowGPUVector(const cv::gpu::GpuMat d_images, vector<cv::gpu::GpuMat> d_flow_u, vector<cv::gpu::GpuMat> d_flow_v);

	/**
	* This function calculates the optical flow of a group of frames which are already loaded in
	* memory. The function will save the results in two GpuMats (x and y component). Each GpuMat
	* will contain all the optical flow mats, that is to say, this option has just two pointers 
	* to optical flow memory rather than a vector with all the mats as calculateOpticalFlowGPUVector().
	* \param d_images  A gpu memory matrix with all the images already loaded
	* \param d_flow_u A gpu memory matrix where every component x of the optical flow will be saved
	* \param d_flow_v A gpu memory matrix where every component y of the optical flow will be saved
	*/
	void CalculateOpticalFlowGPUMat(const cv::gpu::GpuMat d_images, cv::gpu::GpuMat &d_flow_u, cv::gpu::GpuMat &d_flow_v);


	/**
	* This function reads the optical flow from files to pageable memory
	* \param base_name_optical_flow_u Common name of the files where the component x of the optical flow is saved. 
	* The full name should be  as follow: base_name_optical_flow_u + number + .jpg
	* \param base_name_optical_flow_v Common name of the files where the component y of the optical flow is saved. 
	* The full name should be  as follow: base_name_optical_flow_y + number + .jpg
	* \param h_flow_u A cpu memory matrix where the component x of the optical flow will be saved
	* \param h_flow_v A cpu memory matrix where the component y of the optical flow will be saved
	*/
	void ReadFlowFromFilesPageableMemory(string base_name_optical_flow_u, string base_name_optical_flow_v, cv::Mat &h_flow_u, cv::Mat &h_flow_v);

	/**
	* This function reads the optical flow from files to pinned memory
	* \param base_name_optical_flow_u Common name of the files where the component x of the optical flow is saved. 
	* The full name should be  as follow: base_name_optical_flow_u + number + .jpg
	* \param base_name_optical_flow_v Common name of the files where the component y of the optical flow is saved. 
	* The full name should be  as follow: base_name_optical_flow_y + number + .jpg
	* \param h_flow_u A pinned memory matrix where the component x of the optical flow will be saved
	* \param h_flow_v A pinned memory matrix where the component y of the optical flow will be saved
	*/
	void ReadFlowFromFilesPinnedMemory(string base_name_optical_flow_u, string base_name_optical_flow_v, cv::gpu::CudaMem &h_flow_u, cv::gpu::CudaMem &h_flow_v);

	/**
	* This function downloads the optical flow to pageable memory
	* \param d_flow_u A gpu memory matrix with the component x of the optical flow
	* \param d_flow_v A gpu memory matrix with the component y of the optical flow
	* \param h_flow_u A cpu memory matrix where the component x of the optical flow will be saved
	* \param h_flow_v A cpu memory matrix where the component y of the optical flow will be saved
	*/
	void DownloadFlowMatPageableMemory(const cv::gpu::GpuMat d_flow_u,const cv::gpu::GpuMat d_flow_v, cv::Mat &h_flow_u, cv::Mat &h_flow_v);

	/**
	* This function downloads the optical flow to pinned memory.
	* \param d_flow_u A gpu memory matrix with the component x of the optical flow
	* \param d_flow_v A gpu memory matrix with the component y of the optical flow
	* \param h_flow_u A pinned memory matrix where the component x of the optical flow will be saved
	* \param h_flow_v A pinned memory matrix where the component y of the optical flow will be saved
	*/
	void DownloadFlowMatPinnedMemory(const cv::gpu::GpuMat d_flow_u, const cv::gpu::GpuMat d_flow_v, cv::gpu::CudaMem &h_flow_u, cv::gpu::CudaMem &h_flow_v);

	/**
	* This function downloads the optical flow to pageable memory (an array)
	* \param d_flow_u An array of gpu memory matrices with the component x of the optical flow
	* \param d_flow_v An array of gpu memory matrices with the component y of the optical flow
	* \param h_flow_u An array of cpu memory matrices where the component x of the optical flow will be saved
	* \param h_flow_v An array of cpu memory matrices where the component y of the optical flow will be saved
	*/
	void DownloadFlowVectorPageableMemory(vector<cv::gpu::GpuMat> d_flow_u, vector<cv::gpu::GpuMat> d_flow_v, vector<cv::Mat> h_flow_u, vector<cv::Mat> h_flow_v);

	/**
	* This function downloads the optical flow to pinned memory (an array)
	* \param d_flow_u An array of gpu memory matrices with the component x of the optical flow
	* \param d_flow_v An array of gpu memory matrices with the component y of the optical flow
	* \param h_flow_u An array of pinned memory matrices where the component x of the optical flow will be saved
	* \param h_flow_v An array of pinned memory matrices where the component y of the optical flow will be saved
	*/
	void DownloadFlowVectorPinnedMemory(vector<cv::gpu::GpuMat> d_flow_u, vector<cv::gpu::GpuMat> d_flow_v, vector<cv::gpu::CudaMem> h_flow_u, vector<cv::gpu::CudaMem> h_flow_v);


private:

	//! Rows of the image
	int rows_;
	//! Cols of the image
	int cols_;
	//! Total number of frames
	int frames_;
};

/**
* \class Times
*
* \brief This is a class for calculating the times of the whole algorithm
*
* This class provides the user a simple way to calculate the times of the main
* modules of the algorithm. At the same time, it brings a set of methods to print
* the results.
*
* \author Fernando Cobo Aguilera
* \date 30/09/2014
*/

class Times
{
public:

	/**
	* Class constructor
	*/
	Times(void);

	/**
	* Class desctructor
	*/
	~Times(void);

	/**
	* These methods get the exact time when they are called and
	* save the result
	*/
	void StartTime();
	void StopTime();

	/**
	* These methods return the difference of time between a start point
	* and a end point (startTime & stopTime)
	* \return Time difference in miliseconds
	*/
	double GetTimeV1();
	double GetTimeV2();

	/**
	* This method prints the execute time of the cartesians to polars operation
	*/
	void PrintCartToPolarTimer();
	
	/**
	* This method prints the execution time of all the pyramid levels but the first
	*/
	void PrintPyramidalTimer();
	
	/**
	* This method prints the execution total time of the first level of the pyramid
	*/
	void PrintHofTimer();
	
	/**
	* This method prints the execution time of the first level of the pyramid splitted
	* into the main modules
	*/
	void PrintTimesHof();
	
	/**
	* This method prints the execution time of the first level of the pyramid in the gpu
	* version.
	*/	
	void PrintTimesHofGpu();
	
	/**
	* This method prints the execution time of the hof accumulation
	*/	
	void PrintTimesAcumulatedHof();
	
	/**
	* This method prints the final execution time of the program
	*/	
	void PrintProgramTimer();

	/**
	* Set the timers to 0
	*/
	void ResetTimers();

	//! Time to allocate memory in gpu
	double reserve_memory_timer_;
	//! Time to calculate cartesians to polars
	double cartesians_to_polars_timer_;
	//! Time to calculate the weights
	double weights_timer_;
	//! Time to calculate the comparisons of the histograms
	double comparisons_timer_;
	//! Time to calculate the histograms
	double histogram_timer_;
	//! Time to calculate traspose operations
	double transpose_timer_;
	//! Time to calculate hof
	double hof_timer_;
	//! Time to calculate the accumulation of histograms
	double accumulate_timer_;
	//! Time to calculate the normalization of the accumulated histograms
	double normalize_timer_;
	//! Total time of the algorithm
	double program_timer_;
	//! Time to calculate all the pyramid levels but the first
	double pyramidal_hof_timer_;
	//! Time to calculate the upload weights to gpu memory
	double upload_weights_timer_;

private:

//! Variables to measure the time between two points
#if defined _WIN32
	clock_t start_time_;
	clock_t finish_time_;
#else
	timeval start_time_;
	timeval finish_time_;	
#endif
};


/**
* \class SVMPahof
*
* \brief This is a class which is able to train or to classify a group of 
* accumulated histograms with SVM
*
* \author Fernando Cobo Aguilera
* \date 30/09/2014
*/
class SVMPahof{
	
public:

	/**
	* Class constructor
	*/
	SVMPahof(void);

	/**
	* Class constructor
	* \param src_xml Name of the file where the data are saved
	*/
	SVMPahof(string src_xml);

	/**
	* Class destructor
	*/
	~SVMPahof(void);

	/**
	* This function trains a set of data with SVM. The data must be in a matrix where every
	* row is an element.
	* \param data Data that will be trained
	* \param number_of_clases Total number of different classes among the elements
	* \param size_class Total number of elements (rows) of every class
	* \param src_xml Name of the file where the train data will be saved
	*/
	void TrainSVM(cv::Mat data, int number_of_clases, int size_class, string src_xml);

	/**
	* This function classifies a set of data with SVM. The data must be in a matrix where every
	* row is an element.
	* @param data Data that will be classify
	* @param src_xml Name of the file where the train data is saved
	*/
	void ClassifySVM(cv::Mat data, string src_xml);

	/**
	* This function classifies a single element. The train data must be
	* loaded in the constructor.
	* @param data Data that will be trained
	*/
	string PredictSVM(cv::Mat data);

private:

	//! SVM object
	CvSVM svm_;
};


/**
* \class Args
*
* \brief This is a class which helps the main program to check if the command
* line arguments are correct
*
* \author Fernando Cobo Aguilera
* \date 30/09/2014
*/
class Args
{
public:

	/**
	* Class constructor
	*/
    Args();

	/**
	* This function checks if the parameters are correct
	* @param argc Total number of arguments
	* @param argv Array with the arguments
	*/
    static Args Read(int argc, char** argv);

	/**
	* This function shows a help message when the user types the arguments incorrectly
	*/
	static void PrintHelp();

	/**
	* This function checks if the total number of parameters from command line is equal to a number
	* @param argc Total number of parameters
	* @param i Integer that will be compared
	*/
	static void CheckErrors(int argc, int i);

	/**
	* This function checks if a string contains a double or not.
	* \param src String that will be checked
	*/
	static void CheckDouble(char *src);

	/**
	* This function checks if a string contains a integer or not.
	* \param src String that will be checked
	*/
	static void CheckInteger(char * src);

	/**
	* This function checks if a xml file can be opened
	* @param src Name of the xml file
	* @param mode cv::FileStorage mode
	*/
	static void CheckXML(char *src, int mode);

	//! Name of the video
    string src_;

	//! Parameters for the human motion detection
    bool make_gray_;
    bool resize_src_;
    int width_;
	int height_;
    double scale_;
    int nlevels_;
    int gr_threshold_;
    double hit_threshold_;
    bool hit_threshold_auto_;
    int win_width_;
    int win_stride_width_;
	int win_stride_height_;
    bool gamma_corr_;

	//Rows and cols of the images
	int rows_;
	int cols_;

	//! Variable to actiavte the GPU version
	bool gpu_;
	//! Variable to activate the training in SVM
	bool train_;
	//! Variable to activate the classifying in SVM
	bool classify_;

	bool video_;
	bool images_;
	bool xmls_;

	//! Number of frames of the video or total number of images
	int frames_;

	//! Variable to save the accumulated histograms in a xml
	bool xml_;

	//! PaHof parameters
	int len_;
	int step_;
	int n_ang_;
	
	//! Name of the file where the train data is saved 
	string src_svm_;

	//! Variable to show the fps in the video
	bool fps_;
};


/**
* \class HumanMotionDetection
*
* \brief This class is an extension of the App class from Hog.cpp (OpenCV samples). This
* class can detect human motion in a video
*
* \author Fernando Cobo Aguilera
* \date 30/09/2014
*/
class HumanMotionDetection
{
public:

	/**
	* Class constructor
	* @param s Arguments from a Args object
	*/
    HumanMotionDetection(const Args& s);


	/**
	* This function detects human motion in a video and it calculates
	* the optical flow every two detections
	*/
    int Run();

	/**
	* Getter of the component x of the optical flow
	* \return Component x of the optical flow
	*/
	cv::Mat GetOpticalFlowU();

	/**
	* Getter of the component y of the optical flow
	* \return Component y of the optical flow
	*/
	cv::Mat GetOpticalFlowV();

	/**
	* Getter of the total number of detections
	* \return Total number of detections
	*/
	int GetDetections();

	/**
	* These functions calculate the fps of the video in a 
	* particular moment
	*/
	void WorkBegin();
    void WorkEnd();
    string WorkFps() const;



	//cv::Mat show_image_;
	//void PreRun();
	//bool RunTrack(int len);
	//bool RunOneFrame();
	//bool Runv3();
	//cv::Mat GetMagnitudev2(int len);
	//cv::Mat GetAnglev2(int len);

private:

    HumanMotionDetection operator=(HumanMotionDetection&);

    Args args_;
    bool running_;

    bool make_gray_;
    double scale_;
    int gr_threshold_;
    int nlevels_;
    double hit_threshold_;
    bool gamma_corr_;

    int64 work_begin_;
    double work_fps_;

	cv::Mat h_flow_u_, h_flow_v_;

	int detections_;

	 cv::VideoCapture vc2_;
	 cv::gpu::HOGDescriptor gpu_hog2_;
	 cv::gpu::FarnebackOpticalFlow d_flow2_;

	 cv::gpu::GpuMat last_frame_;
	 cv::Rect last_rect_;
};


#endif //CUPAHOF_UTILITIES_H