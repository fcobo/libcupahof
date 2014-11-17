#include "utilities.h"
#include "pahof.h"
#include "parameters.h"

/**
* This file contains the main program of PaHOF algorithm where the user can calculate
* PaHOF in the CPU or cuPaHOF in the GPU.
*
* If you use a video, you need to decide the size of the images that will be used in the algorithm.
* By default: 128x128. Check parameters.h out to change these values manually.
*
*
* The number of bins for Optical Flow angle quantization is 8 by default.
* The magnitude intervals for quantization are 4 by default: {0.5, 1.5, 2.5, 999999}.

*								-----ADVERTISEMENT-----
*
* If you want to change these two last parameters, you have to do it manually. Futhermore,
* you will have to change the value of VECTOR_N in parameters.h
*/


int main(int argc, char** argv){

	cout << "\t\t---cuPaHOF algorithm---" << endl;


	//Subdivisions to the components of the optical flow
	uchar m[3][2] = {{4, 8}, {2, 4}, {1, 2}};
	cv::Mat xytiles(3,2,CV_8U,m);

	//Number of frames to accumulate
	int len; //20 by default;

	//Every 'step_' frames a descriptor is computed
	int step; //2 by default;

	int frames; //0

	//Magnitude intervals for quantization
	float nMag[] = {0.5, 1.5, 2.5, 999999}; //n_mags = 4

	//Number of bins for Optical Flow angle quantization
	int nAng;  //n_ang = 8
	

	//Detecting motion
	 Args args = Args::Read(argc, argv);

	 if(args.gpu_){

		//Initialize CUDA graphic card
		cout << "Initializing CUDA graphic card..."<<endl;

		cudaError_t error;

		int ndevices = cv::gpu::getCudaEnabledDeviceCount();
    
	    if(ndevices == 0){
	
          cerr << "No CUDA-capable devices were detected by the installed CUDA driver" << endl;
          cout << "Press to end";
          getchar();
          exit(EXIT_FAILURE);
		}

		if( cudaSuccess != (error = cudaSetDevice(0)) ){

          cerr << "There was a problem during GPU initializaction.";

          switch (error)
          {
          case 10:
            cerr << "The device which has been supplied by the user does not correspond to a valid CUDA device." 
                 << " Try to change cudaSetDevice() with another value."<<endl;
            break;
          default:
            break;
          }
    
        cout << "Press to end";
        getchar();
        exit(EXIT_FAILURE);
		}
		cv::gpu::printShortCudaDeviceInfo(0);
	 }

	 frames = args.frames_;
	 len = args.len_;
	 step = args.step_;
	 nAng = args.n_ang_;	

	PAHof pahofcuda(nAng, nMag, len, step, frames, ROWS, COLS, xytiles, (int)sizeof(nMag)/sizeof(float));
	cv::Mat pahof;

	cv::Mat h_optical_flow_u;
	cv::Mat h_optical_flow_v;

	if(args.xmls_){

		string basename_u = args.src_ + "U";
	    string basename_v = args.src_ + "V";  

		pahofcuda.LoadOpticalFlowFromFiles(basename_u,basename_v,h_optical_flow_u,h_optical_flow_v);

		if(args.gpu_)
			pahof = pahofcuda.PAHofGpu(h_optical_flow_u,h_optical_flow_v,true);
		else
			pahof = pahofcuda.PAHofCpu(h_optical_flow_u,h_optical_flow_v,true);
	}
	else{

		if(args.images_){

			string basename =  args.src_; 

			pahofcuda.CalculateOpticalFlowFromImages(basename,h_optical_flow_u,h_optical_flow_v);

			if(args.gpu_)
			 pahof = pahofcuda.PAHofGpu(h_optical_flow_u,h_optical_flow_v,true);
			else
			 pahof = pahofcuda.PAHofCpu(h_optical_flow_u,h_optical_flow_v,true);
		}
		else{

			if(args.video_){

				HumanMotionDetection human_motion(args);

				human_motion.Run();

					 if(human_motion.GetDetections()<len){

						 cerr << "Insufficient detections"<< endl;
						exit(EXIT_FAILURE);
					 }

				pahofcuda.SetFrames(human_motion.GetDetections());

				if(args.gpu_){
					pahof = pahofcuda.PAHofGpu(human_motion.GetOpticalFlowU(),human_motion.GetOpticalFlowV(),true);}
				else
				    pahof = pahofcuda.PAHofCpu(human_motion.GetOpticalFlowU(),human_motion.GetOpticalFlowV(),true);
			}
		}
	}


	if(args.xml_){

		cout << "Saving Pahof in file...";
		cv::FileStorage fs("Pahof.xml", cv::FileStorage::WRITE);
		fs << "pahof" << pahof;
		fs.release();
		cout << "Done!"<<endl;
	}

	if(args.classify_){

		SVMPahof svmpahof;	 
		svmpahof.ClassifySVM(pahof, args.src_svm_);
	}


	cout << "Press to end...";
	getchar();
	return 0;
}