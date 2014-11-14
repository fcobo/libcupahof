/**
* \file pahof.cpp
* This file contains the implementation of the main classes of the algorithm where
* PaHOF is calculated
*
* @author Fernando Cobo Aguilera
* @date October 2014
*/

#include "pahof.h"

Hof::Hof(){
}

Hof::Hof(int rows, int cols, int frames, int ang,float *mag, int size_mag)
{
	rows_ = rows;
	cols_ = cols;
	frames_ = frames;

	n_ang_ = ang;
	mags_ = mag;
	n_mags_ = size_mag;

}

Hof::Hof(int rows, int cols, int frames){
		
	rows = rows;
	cols = cols;
	frames = frames;
}

Hof::~Hof(void)
{
}


void Hof::CalculateCartToPolarGpuPinnedMemory(const cv::Mat h_flow_u,const cv::Mat h_flow_v, cv::gpu::CudaMem &h_magnitude, cv::gpu::CudaMem &h_angle){

	cv::gpu::GpuMat d_flow_u;
	cv::gpu::GpuMat d_flow_v;

	cv::gpu::GpuMat d_angle;
	cv::gpu::GpuMat d_magnitude;

	cv::Mat h_magnitude_header;
	cv::Mat h_angle_header;

	//Optical flow is uploaded to gpu memory
	d_flow_u.upload(h_flow_u);
	d_flow_v.upload(h_flow_v);

	//Conversion is calculated
	cv::gpu::cartToPolar(d_flow_u,d_flow_v,d_magnitude,d_angle);

	//Pinned memory is allocated
	h_magnitude.create(cv::Size(d_magnitude.cols,d_magnitude.rows),d_magnitude.type());
	h_angle.create(cv::Size(d_angle.cols,d_angle.rows),d_angle.type());

	h_magnitude_header = h_magnitude.createMatHeader();
	h_angle_header = h_angle.createMatHeader();
	//Polar coordinates are downloaded to pinned memory (cpu)
	d_magnitude.download(h_magnitude_header);
	d_angle.download(h_angle_header);

	d_angle.release();
	d_magnitude.release();
}

void Hof::CalculateCartToPolarCpuPageableMemory(const cv::Mat h_flow_u,const cv::Mat h_flow_v, cv::Mat &h_magnitude, cv::Mat &h_angle){
	cv::cartToPolar(h_flow_u,h_flow_v, h_magnitude, h_angle);
}

void Hof::SetFrames(int n_frames){

	frames_ = n_frames;
}


cv::Mat Hof::ComputeHofCpu(int x_tiles, int y_tiles, cv::Mat h_angles, cv::Mat h_magnitudes, int n_histograms){
	
	cv::Mat hof;
	cv::Mat weight;
	cv::Mat sumValAng;
	cv::Mat hofHeader;
	cv::Mat transposeHof;

	//Masks are calculated
	CalculateMasks(x_tiles,y_tiles);

	timer.StartTime();
	//Weights are calculated
	weight = CalculateWeights(x_tiles,y_tiles);
	timer.StopTime();
	timer.weights_timer_ = timer.GetTimeV2();

	for(int i=0;i<n_mags_;i++)
	 mod_mask_.push_back(cv::Mat::zeros(rows_,cols_,CV_8U));
	
	hof =  cv::Mat::zeros(x_tiles*y_tiles*n_ang_*n_mags_,n_histograms,CV_16U);


	//Main Loop
	//Histograms calculation
	for(int i=0;i< n_histograms;i++){
		
		timer.StartTime();
		//Comparations are executed
		sumValAng = CalculateHistogramComparisons(h_angles(cv::Rect(0,i*rows_,cols_,rows_)),
										     h_magnitudes(cv::Rect(0,i*rows_,cols_,rows_)));
		timer.StopTime();
		timer.comparisons_timer_ += timer.GetTimeV2();


		hofHeader = hof(cv::Rect(i,0,1,hof.rows));
		timer.StartTime();
		//Dot products are executed
		CalculateHistogramDotProductCpu(x_tiles,y_tiles,sumValAng,weight,hofHeader);	
		timer.StopTime();
		timer.histogram_timer_ += timer.GetTimeV2();
	}

	cv::transpose(hof,transposeHof);
	
	return transposeHof;
}

cv::Mat Hof::ComputeHofGpu(int x_tiles, int y_tiles, cv::Mat h_angles, cv::Mat h_magnitudes, int n_histograms){


	timer.StartTime();

	cv::gpu::GpuMat weight;
	cv::Mat hof;
	cv::gpu::CudaMem header_weight;

	//With the buffer we avoid allocating too much gpu memory
	BufferHof buffer;

	//Streams for asynchronous calls
	buffer.streams = new cudaStream_t[n_histograms];
	buffer.d_sum_val_ang = new cv::gpu::GpuMat[n_histograms];
	buffer.h_sum_val_ang = new cv::gpu::CudaMem[n_histograms];

	for(int i=0; i<n_histograms; i++)
		cudaStreamCreate(&buffer.streams[i]);
	
	for(int i=0; i<n_histograms; i++)
	buffer.h_sum_val_ang[i].create(rows_*n_ang_*n_mags_,cols_,CV_8U);

	//It's not needed to allocate n_histograms d_sum_val_ang. With 4 is enough (allocate memory consumes a lot of time)
	//It may be changed to n_histograms if results are not correct. The overlapping is different in every GPU.
	for(int i=0; i<4; i++)
	 buffer.d_sum_val_ang[i].create(rows_*n_ang_*n_mags_,cols_,CV_8U);


	buffer.poshisto.create(rows_,cols_,CV_8U);
	buffer.poshi.create(buffer.poshisto.rows,buffer.poshisto.cols,CV_8U);
	buffer.h_histogram.create(n_histograms,n_ang_*n_mags_*x_tiles*y_tiles,CV_16U);
	buffer.d_histogram.create(n_histograms,n_ang_*n_mags_*x_tiles*y_tiles,CV_16U);

	for(int i=0;i<n_mags_;i++)
	 mod_mask_.push_back(cv::Mat::zeros(rows_,cols_,CV_8U));

	timer.StopTime();
	timer.reserve_memory_timer_ = timer.GetTimeV2();
	
	//Masks are calculated
	CalculateMasks(x_tiles,y_tiles);

	timer.StartTime();
	//Weights are calculated
	header_weight = CalculateWeightsCudaMem(x_tiles,y_tiles);
	timer.StopTime();
	timer.weights_timer_ = timer.GetTimeV2();

	cudaEvent_t start,stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	timer.StartTime();
	//Weights are uploaded
	weight.upload(header_weight);
	timer.StopTime();
	timer.upload_weights_timer_ = timer.GetTimeV2();
	
	hof =  cv::Mat::zeros(n_histograms,x_tiles*y_tiles*n_ang_*n_mags_,CV_16U);

	cudaEventRecord(start,0);
	//Main Loop
	/* In this loop, both CPU and GPU work together due to asynchronous calls
	For the CPU, cudaMemcpyAsync and CalculateHistogramGpu dont exist
	*/
	for(int i=0;i< n_histograms;i++){

		timer.StartTime();

		//Comparations are calculated in CPU
		CalculateHistogramComparisonsCudaMem(h_angles(cv::Rect(0,i*rows_,cols_,rows_)),
												 h_magnitudes(cv::Rect(0,i*rows_,cols_,rows_)),
												 i,buffer);
		timer.StopTime();
		timer.comparisons_timer_ += timer.GetTimeV2();

		//Asynchronous copy to gpu memory (non-blocking)
		//i%4 should be changed depending on the memory allocated on d_sum_val_ang. In this case, 4 pointers is enough.
		cudaMemcpyAsync(buffer.d_sum_val_ang[i%4].data,buffer.h_sum_val_ang[i].data,rows_*n_ang_*n_mags_*cols_*sizeof(uchar),cudaMemcpyHostToDevice,buffer.streams[i]);
		
		//GPU kernel to calculate Dot product (non-blocking)
		CalculateHistogramDotProductGpu(x_tiles,y_tiles,buffer.d_sum_val_ang[i%4],weight,buffer,i);
	}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	timer.histogram_timer_ = elapsedTime;

	buffer.d_histogram.download(buffer.h_histogram);
	
	buffer.d_histogram.release();

	//GPU memory is released
	for(int i=0; i< n_histograms; i++){
		buffer.d_sum_val_ang[i].release();
		buffer.h_sum_val_ang[i].release();
	}

	return buffer.h_histogram;
}

cv::Mat Hof::PyramidalHof(cv::Mat hof, int x_tiles, int y_tiles){

	cv::Mat nextHof;

    nextHof =  cv::Mat::zeros(hof.rows,x_tiles*y_tiles*n_ang_*n_mags_,CV_16U);

    timer.StartTime();

    for(int n=0;n< nextHof.rows ;n++){ //Total frames

        for(int i=0; i< x_tiles ; i++){

        for(int j=0; j< y_tiles; j++){
        
                for(int k=0; k< n_ang_*n_mags_; k++){

                    nextHof.at<short int>(n, k + (n_ang_*n_mags_)*(j + (y_tiles*i))) = 
                                                                   hof.at<short int>(n,k + ((j + (y_tiles*2*i))*n_ang_*n_mags_*2)) + 
                                                                   hof.at<short int>(n,k + ((j + (y_tiles*2*i))*n_ang_*n_mags_*2)  + (n_ang_*n_mags_)) + 
                                                                   hof.at<short int>(n,k + ((j + (y_tiles*2*i))*n_ang_*n_mags_*2)  + (n_ang_*n_mags_)*y_tiles*2) + 
                                                                   hof.at<short int>(n,k + ((j + (y_tiles*2*i))*n_ang_*n_mags_*2)  + (n_ang_*n_mags_)*y_tiles*2 + (n_ang_*n_mags_));
                }
        }
        }
    }
    timer.StopTime();
    timer.pyramidal_hof_timer_ += timer.GetTimeV2();

    return nextHof;
}



void Hof::CalculateMasks(int x_tiles, int y_tiles){

	mask_x_ = cv::Mat::zeros(cols_,x_tiles,CV_32F);
			int stepx= (int)ceil(cols_/x_tiles);
			  for(int x=0;x<x_tiles;x++)
				  mask_x_.col(x).rowRange(x*stepx,min((x+1)*stepx,cols_)) = 1;

    cv::transpose(mask_x_,mask_x_);
 
	mask_y_ = cv::Mat::zeros(rows_,y_tiles,CV_32F);
			int stepy= (int)ceil(rows_/y_tiles);
			  for(int y=0;y<y_tiles;y++)
				  mask_y_.col(y).rowRange(y*stepy,min((y+1)*stepy,rows_)) = 1;
}
	
cv::Mat Hof::CalculateWeights(int x_tiles, int y_tiles){

	 cv::Mat weight;
	weight.create(rows_*x_tiles*y_tiles,cols_,0);
	 
	 for(int x=0; x<x_tiles;x++)
	   for(int y=0;y<y_tiles;y++)
		   weight(cv::Rect(0,rows_*(y + (x*y_tiles)),cols_,rows_))  = (( mask_y_.col(y) * mask_x_.row(x)) > 0) * _1_255;
	   
	 return weight;
}

cv::gpu::CudaMem Hof::CalculateWeightsCudaMem(int tidx, int tidy){

	cv::gpu::CudaMem weight;
	weight.create(rows_*tidx*tidy,cols_,0);

	cv::Mat header = weight.createMatHeader();
	 
	 for(int x=0; x<tidx;x++)
	   for(int y=0;y<tidy;y++)
		  header(cv::Rect(0,rows_*(y + (x*tidy)),cols_,rows_))  = (( mask_y_.col(y) * mask_x_.row(x)) > 0) * _1_255;
	   
	 return weight;
}


cv::Mat Hof::CalculateHistogramComparisons(cv::Mat h_angle, cv::Mat h_magnitude){

	cv::Mat poshisto(rows_,cols_,CV_8U);
	cv::Mat poshi(rows_,cols_,CV_8U);
	cv::Mat sumValAng(rows_*n_ang_*n_mags_,cols_,CV_8U);

	for(int i=0;i<rows_;i++)
	 for(int j=0;j<cols_;j++){
	  poshisto.at<uchar>(i,j) = (uchar)floor((h_angle.at<float>(i,j)+M_PI)*(n_ang_-1E-20)/(2.0*M_PI));
	  poshisto.at<uchar>(i,j) = 1 + poshisto.at<uchar>(i,j) % n_ang_;
	 }

	mod_mask_[0] = (mags_[0] >= h_magnitude) * _1_255;

	for(int k=0; k<n_mags_-1; k++)
	 mod_mask_[k+1] = ((mags_[k] < h_magnitude) & (mags_[k+1] >= h_magnitude)) * _1_255;

	for(int i=0;i<n_ang_;i++){
		poshi = (poshisto == i+1) * _1_255;
	
		for(int k=0; k< n_mags_; k++)
			multiply(poshi,mod_mask_[k],sumValAng(cv::Rect(0,(i*n_mags_ +k)*rows_,cols_,rows_)));					
	}

	return sumValAng;
}

void Hof::CalculateHistogramComparisonsCudaMem(cv::Mat h_angle, cv::Mat h_magnitude, int index,BufferHof buffer){

	for(int i=0;i<rows_;i++)
	 for(int j=0;j<cols_;j++){
	  buffer.poshisto.at<uchar>(i,j) = (uchar)floor((h_angle.at<float>(i,j)+M_PI)*(n_ang_-1E-20)/(2.0*M_PI));
	  buffer.poshisto.at<uchar>(i,j) = 1 + buffer.poshisto.at<uchar>(i,j) % n_ang_;
	 }

	mod_mask_[0] = (mags_[0] >= h_magnitude) * _1_255;

	for(int k=0; k<n_mags_-1; k++)
	 mod_mask_[k+1] = ((mags_[k] < h_magnitude) & (mags_[k+1] >= h_magnitude)) * _1_255;

	cv::Mat headersumValAng = buffer.h_sum_val_ang[index].createMatHeader();

	for(int i=0;i<n_ang_;i++){
		buffer.poshi = (buffer.poshisto == i+1) * _1_255;
	
		for(int k=0; k< n_mags_; k++)
			multiply(buffer.poshi,mod_mask_[k],headersumValAng(cv::Rect(0,(i*n_mags_ +k)*rows_,cols_,rows_)));					
	}
}


void Hof::CalculateHistogramDotProductCpu(int x_tiles, int y_tiles, cv::Mat h_sum_val_ang, cv::Mat h_weight, cv::Mat h_histogram){

	int s=0;

	for(int x=0; x<x_tiles;x++)
	 for(int y=0;y<y_tiles;y++)
	  for(int i=0;i<n_ang_;i++)
	   for(int k=0;k<n_mags_;k++){
		 h_histogram.at<short int>(s,0) = (short int)h_sum_val_ang(cv::Rect(0,(i*n_mags_ +k)*rows_,cols_,rows_)).dot(h_weight(cv::Rect(0,rows_*(x*y_tiles + y),cols_,rows_)));
		 s++;
	   }
}

void Hof::CalculateHistogramDotProductGpu(int x_tiles, int y_tiles, cv::gpu::GpuMat d_sum_val_ang,cv::gpu::GpuMat d_weight, BufferHof buffer, int index){

	for(int x=0; x<x_tiles;x++)
		for(int y=0;y<y_tiles;y++)
		 DotProduct(d_sum_val_ang,d_weight(cv::Rect(0,rows_*(y_tiles*x +y),cols_,rows_)),buffer.streams[index],buffer.d_histogram(cv::Rect((y + x*y_tiles)*(n_ang_*n_mags_),index,n_ang_*n_mags_,1)));	  
		

}

int Hof::GetNAng(){
	return n_ang_;
}

float * Hof::GetMags(){
	return mags_;
}

int Hof::GetNMags(){
	return n_mags_;
}

AHof::AHof(void){
}

AHof::AHof(int leng, int st){

	len_ = leng;
	step_ = st;
}

AHof::~AHof(void){
}

cv::Mat AHof::ComputeAhof(cv::Mat h_hof){
	
	int slen = h_hof.rows;
	int final_ix = cv::max(1,slen-len_);
	int contador=1;

	cv::Mat  aux;
	cv::Mat aHOF;
	
	int *ptraHOF;
	const unsigned short *ptrHOF;

	aHOF.create(ceil((float)final_ix/step_),h_hof.cols,CV_32S);
	ptraHOF = aHOF.ptr<int>(0);

    for (int ix=0 ; ix<final_ix; ix= ix + step_){	
	  aux = h_hof.rowRange(ix,min(slen,ix+len_-1));
	  ptrHOF = aux.ptr<unsigned short>(0);
	  for(int i=0; i< aux.cols ;i++){
		    ptraHOF[i]=0;
		for(int j=0; j< aux.rows; j++){
			ptraHOF[i] += (int)ptrHOF[i+ j*aux.cols];
		}
	  }
	  ptraHOF = aHOF.ptr<int>(contador);
	  contador++;
    }

	return aHOF;
}

cv::Mat AHof::ComputeAhofOptimized(cv::Mat h_hof){

	int slen = h_hof.rows;
	int final_ix = cv::max(1,slen-len_);
	int contador = 0;

	cv::Mat  aux, aux1;
	cv::Mat aHOF;

	int *ptraHOF;
	int *ptraHOF2;
	const unsigned short *ptrHOF;
	const unsigned short *ptrHOF2;

	aux = h_hof.rowRange(0,min(slen,len_-1)).colRange(0,h_hof.cols);

	ptrHOF = aux.ptr<unsigned short>(0);

	aHOF.create((int)final_ix/step_ +1,h_hof.cols,CV_32S);

	ptraHOF = aHOF.ptr<int>(0);

	for(int i=0; i< aux.cols ;i++){
		    ptraHOF[i]=0;
		for(int j=0; j< aux.rows; j++){
			ptraHOF[i] += (int)ptrHOF[i+ j*aux.cols];
		}	 		
	}

	ptraHOF  = aHOF.ptr<int>(contador+1);
	ptraHOF2 = aHOF.ptr<int>(contador);
	contador++;

	for (int ix=step_ ; ix<final_ix; ix= ix + step_){
		
		aux1 = h_hof.rowRange(ix,min(slen,ix+len_-1)).colRange(0,h_hof.cols);
		ptrHOF2 = aux1.ptr<unsigned short>(0);

		for(int i=0; i< aHOF.cols ;i++){
				  ptraHOF[i] = ptraHOF2[i];

			for(int j= 0; j < step_; j++)
					ptraHOF[i] -= (int)ptrHOF[i + j*aHOF.cols];

			for(int j= aux1.rows - step_ ; j<aux1.rows;j++)
					ptraHOF[i] += (int)ptrHOF2[i + j*aHOF.cols];		
		}

		if(contador+1 <= final_ix/step_){
			ptraHOF  = aHOF.ptr<int>(contador+1);
			ptraHOF2 = aHOF.ptr<int>(contador);
			contador++;
		}

		ptrHOF = h_hof.rowRange(ix,min(slen,ix+len_-1)).colRange(0,h_hof.cols).ptr<unsigned short>(0);
	}

	return aHOF;
}

cv::Mat AHof::NormalizeAhof(int nMags, cv::Mat h_ahof){

	cv::Mat aux;

	h_ahof += 1;

	h_ahof.convertTo(h_ahof,CV_32F);

	for(int i=0; i<h_ahof.rows; i++){
		for(int j=0; j< h_ahof.cols; j=j+nMags){

				aux = h_ahof.row(i).colRange(j,j+nMags);
				aux = aux / sum(aux)[0];
		}
	}	

    return h_ahof;
}

PAHof::PAHof(){
}

PAHof::PAHof(int ang, float *mag, int leng, int st, int frames, int rows, int cols,cv::Mat xytiles, int size_mag){

	 rows_ = rows;
	 cols_ = cols;
	 frames_ = frames;

	 xytiles.copyTo(x_y_tiles_);

	 sums_ = new int[x_y_tiles_.rows];

	 if(frames_ == 0 || cols_ == 0 || rows_ == 0 || sizeof(mag) == 0){

		 cerr << "Error PAHof: valor inicializado a cero" << endl;
		 exit(-1);
	 }

	 //Initialize objects
	 optical_flow_management_ = OpticalFlowManagement( rows_, cols_, frames_);
	 histograms_optical_flow_ = Hof(rows_, cols_, frames_, ang, mag, size_mag);
	 accumulated_histograms_optical_flow_ = AHof( leng, st );
}

PAHof::~PAHof(void){

}


void PAHof::CalculateOpticalFlowFromImages(string base_name,cv::Mat &h_flow_u, cv::Mat &h_flow_v){

	cv::gpu::GpuMat d_flow_u;
	cv::gpu::GpuMat d_flow_v;

	 //Images are loaded from memory
	 cv::gpu::GpuMat d_images = optical_flow_management_.LoadImagesFromCpuToGpuPageableMemory(base_name);

	 if(d_images.rows != rows_*frames_ || d_images.cols != cols_){
		 
		 cerr << "Error PAHof: imágenes cargadas a memoria gpu incorrectamente" << endl;
		 exit(-1);
	 }
	 else{
		 cout << "Imagenes cargadas en memoria gpu!"<< endl;
	 }


	 //Optical flow is calculated
	 optical_flow_management_.CalculateOpticalFlowGPUMat(d_images,d_flow_u,d_flow_v);

	 if(d_flow_u.cols != d_images.cols || d_flow_u.rows != d_images.rows ||
		d_flow_v.cols != d_images.cols || d_flow_v.rows != d_images.rows)
	 {

		cerr << "Error PAHof: flujo optico no calculado correctamente" << endl;
		exit(-1);
	 }
	 else{
		 cout << "Flujo optico calculado!" << endl;
	 }


	 //Optical flow is downloaded to cpu memory
	 optical_flow_management_.DownloadFlowMatPageableMemory(d_flow_u,d_flow_v,h_flow_u,h_flow_v);

	  if(d_flow_u.cols != h_flow_u.cols || d_flow_u.rows != h_flow_u.rows ||
		d_flow_v.cols != h_flow_v.cols || d_flow_v.rows != h_flow_v.rows)
	 {

		cerr << "Error PAHof: flujo optico no descargado correctamente" << endl;
		exit(-1);
	 }
	  else{
		  cout << "Flujo optico descargado!" << endl;
	  }

	 //Release gpu memory
	 d_images.release();
	 d_flow_u.release();
	 d_flow_v.release();
}

void PAHof::LoadOpticalFlowFromFiles(string base_name_of_u,string base_name_of_v,cv::Mat &h_flow_u, cv::Mat &h_flow_v){

	//Images are loaded from XML files
	optical_flow_management_.ReadFlowFromFilesPageableMemory(base_name_of_u,base_name_of_v,h_flow_u,h_flow_v);
}



cv::Mat PAHof::PAHofCpu(cv::Mat h_flow_u, cv::Mat h_flow_v, bool acc_optimized, int pyramid_levels){

	if(pyramid_levels > x_y_tiles_.rows){
		cerr << "Error PAHof: demasiados niveles de piramide" << endl;
		exit(-1);
	}

	histograms_optical_flow_.SetFrames(frames_);
	histograms_optical_flow_.timer.ResetTimers();

	cv::Mat h_histograms;
	cv::Mat h_acc_histograms;
	cv::Mat h_norm_acc_histograms;
	cv::Mat h_pyramid_norm_acc_histograms;

	cv::Mat h_optical_flow_magnitude;
	cv::Mat h_optical_flow_angle;

	Times timesMain;

	timesMain.StartTime();
	histograms_optical_flow_.CalculateCartToPolarCpuPageableMemory(h_flow_u, h_flow_v, h_optical_flow_magnitude, h_optical_flow_angle);
	timesMain.StopTime();
	timesMain.cartesians_to_polars_timer_ = timesMain.GetTimeV2();

	if(h_optical_flow_magnitude.rows == 0 || h_optical_flow_magnitude.cols == 0 ||
	   h_optical_flow_angle.rows == 0 || h_optical_flow_angle.cols == 0){

		   cerr << "Error PAHof: conversion de Cartesianas a Polares incorrecta" << endl;
		   exit(-1);
	}
	else{
		cout << "Conversion de Cartesianas a Polares realizada!" << endl;
	}

	for(int i=0; i<x_y_tiles_.rows; i++){

		if(i < pyramid_levels){
		 timesMain.StartTime();
		 h_histograms = histograms_optical_flow_.ComputeHofCpu((int)x_y_tiles_.at<uchar>(i,0),
													          (int)x_y_tiles_.at<uchar>(i,1),
															  h_optical_flow_angle,
															  h_optical_flow_magnitude,frames_-1);
		 timesMain.StopTime();
		 timesMain.hof_timer_ = timesMain.GetTimeV2();
		}
		else{
		 timesMain.StartTime();
		 h_histograms = histograms_optical_flow_.PyramidalHof(h_histograms,
														     (int)x_y_tiles_.at<uchar>(i,0),
													         (int)x_y_tiles_.at<uchar>(i,1));
		 timesMain.StopTime();
		 timesMain.pyramidal_hof_timer_ += timesMain.GetTimeV2();
		}

		sums_[i] = (int)cv::sum(h_histograms)[0];
		cout << "HOF -> Nivel " << i << " calculado!" << endl;

		if(!acc_optimized)
	     h_acc_histograms = accumulated_histograms_optical_flow_.ComputeAhof(h_histograms);
		else
		 h_acc_histograms = accumulated_histograms_optical_flow_.ComputeAhofOptimized(h_histograms);

		cout << "AHOF -> Nivel " << i << " calculado!" << endl;

	    h_norm_acc_histograms = accumulated_histograms_optical_flow_.NormalizeAhof(histograms_optical_flow_.GetNMags(),h_acc_histograms);

		if(i!=0)
         cv::hconcat(h_pyramid_norm_acc_histograms,h_norm_acc_histograms,h_pyramid_norm_acc_histograms);
		else
		  h_norm_acc_histograms.copyTo(h_pyramid_norm_acc_histograms);

	}


	cout << "--Check Sums--" << endl;
	for(int i=0; i<x_y_tiles_.rows; i++){	
		cout << "Nivel " << i <<": " << sums_[i] << endl;

		if(i>0){
			if(sums_[i] != sums_[i-1]){
				cout << "Error en el calculo HOF" << endl;
				break;
			}
		}
	}

	timesMain.PrintCartToPolarTimer();
	histograms_optical_flow_.timer.PrintTimesHof();
	timesMain.PrintPyramidalTimer();
	timesMain.PrintHofTimer();


	return h_pyramid_norm_acc_histograms;
}

cv::Mat PAHof::PAHofGpu(cv::Mat h_flow_u, cv::Mat h_flow_v, bool acc_optimized, int pyramid_levels){

	if(pyramid_levels > x_y_tiles_.rows){
		cerr << "Error PAHof: demasiados niveles de piramide" << endl;
		exit(-1);
	}
	
	histograms_optical_flow_.SetFrames(frames_);
	histograms_optical_flow_.timer.ResetTimers();

	cv::Mat h_histograms;
	cv::Mat h_acc_histograms;
	cv::Mat h_norm_acc_histograms;
	cv::Mat h_pyramid_norm_acc_histograms;

	cv::gpu::CudaMem h_optical_flow_magnitude,h_optical_flow_angle;

	Times timesMain;

	timesMain.StartTime();
	histograms_optical_flow_.CalculateCartToPolarGpuPinnedMemory(h_flow_u, h_flow_v, h_optical_flow_magnitude, h_optical_flow_angle);
	timesMain.StopTime();
	timesMain.cartesians_to_polars_timer_ = timesMain.GetTimeV2();


	if(h_optical_flow_magnitude.rows == 0 || h_optical_flow_magnitude.cols == 0 ||
	   h_optical_flow_angle.rows == 0 || h_optical_flow_angle.cols == 0){

		   cerr << "Error PAHof: conversion de Cartesianas a Polares incorrecta" << endl;
		   exit(-1);
	}
	else{
		cout << "Conversion de Cartesianas a Polares realizada!" << endl;
	}


	for(int i=0; i<x_y_tiles_.rows; i++){


		if(i < pyramid_levels){
		 timesMain.StartTime();
		 h_histograms = histograms_optical_flow_.ComputeHofGpu((int)x_y_tiles_.at<uchar>(i,0),
													         (int)x_y_tiles_.at<uchar>(i,1),
															 h_optical_flow_angle,
															 h_optical_flow_magnitude,frames_-1);
		 timesMain.StopTime();
		 timesMain.hof_timer_ = timesMain.GetTimeV2();
		}
		else{
		timesMain.StartTime();
		 h_histograms = histograms_optical_flow_.PyramidalHof(h_histograms,
														     (int)x_y_tiles_.at<uchar>(i,0),
													         (int)x_y_tiles_.at<uchar>(i,1));
		 timesMain.StopTime();
		 timesMain.pyramidal_hof_timer_ += timesMain.GetTimeV2();
		}

		sums_[i] = (int)cv::sum(h_histograms)[0];
		cout << "HOF -> Nivel " << i << " calculado!" << endl;

	    if(!acc_optimized)
	     h_acc_histograms = accumulated_histograms_optical_flow_.ComputeAhof(h_histograms);
		else
		 h_acc_histograms = accumulated_histograms_optical_flow_.ComputeAhofOptimized(h_histograms);

		cout << "AHOF -> Nivel " << i << " calculado!" << endl;

	    h_norm_acc_histograms = accumulated_histograms_optical_flow_.NormalizeAhof(histograms_optical_flow_.GetNMags(),h_acc_histograms);

		if(i!=0)
         cv::hconcat(h_pyramid_norm_acc_histograms,h_norm_acc_histograms,h_pyramid_norm_acc_histograms);
		else
		  h_norm_acc_histograms.copyTo(h_pyramid_norm_acc_histograms);

	}

	cout << "--Check Sums--" << endl;
	for(int i=0; i<x_y_tiles_.rows; i++){	
		cout << "Nivel " << i <<": " << sums_[i] << endl;

		if(i>0){
			if(sums_[i] != sums_[i-1]){
				cout << "Error en el calculo HOF" << endl;
				break;
			}
		}
	}

	timesMain.PrintCartToPolarTimer();
	histograms_optical_flow_.timer.PrintTimesHofGpu();
	timesMain.PrintPyramidalTimer();
	timesMain.PrintHofTimer();


	return h_pyramid_norm_acc_histograms;

}

void PAHof::SetFrames(int frames){

	frames_ = frames;
}
