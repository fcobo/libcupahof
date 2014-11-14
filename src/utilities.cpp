/**
* \file utilities.cpp
* This source file contains the implementation of classes that are defined in
* utilities.h

* @author Fernando Cobo Aguilera
* @date October 2014
*/


#include "utilities.h"

void Utilities::WriteMat(cv::Mat const& mat, const char* file_name, const char* var_name, bool bgr_to_rgb)
{
   int textLen = 116;
   char* text;
   int subsysOffsetLen = 8;
   char* subsysOffset;
   int verLen = 2;
   char* ver;
   char flags;
   int bytes;
   int padBytes;
   int bytesPerElement;
   int i,j,k,k2;
   bool doBgrSwap;
   char mxClass;
   int32_t miClass;
   uchar const* rowPtr;
   uint32_t tmp32;
   FILE* fp;

   // Matlab constants.
   const uint16_t MI = 0x4d49; // Contains "MI" in ascii.
   const int32_t miINT8 = 1;
   const int32_t miUINT8 = 2;
   const int32_t miINT16 = 3;
   const int32_t miUINT16 = 4;
   const int32_t miINT32 = 5;
   const int32_t miUINT32 = 6;
   const int32_t miSINGLE = 7;
   const int32_t miDOUBLE = 9;
   const int32_t miMATRIX = 14;
   const char mxDOUBLE_CLASS = 6;
   const char mxSINGLE_CLASS = 7;
   const char mxINT8_CLASS = 8;
   const char mxUINT8_CLASS = 9;
   const char mxINT16_CLASS = 10;
   const char mxUINT16_CLASS = 11;
   const char mxINT32_CLASS = 12;
   const uint64_t zero = 0; // Used for padding.

   fp = fopen( file_name, "wb");

   if( fp == 0 )
      return;

   const int rows = mat.rows;
   const int cols = mat.cols;
   const int chans = mat.channels();

   doBgrSwap = (chans==3) && bgr_to_rgb;

   // I hope this mapping is right :-/
   switch( mat.depth() )
   {
   case CV_8U:
      mxClass = mxUINT8_CLASS;
      miClass = miUINT8;
      bytesPerElement = 1;
      break;
   case CV_8S:
      mxClass = mxINT8_CLASS;
      miClass = miINT8;
      bytesPerElement = 1;
      break;
   case CV_16U:
      mxClass = mxUINT16_CLASS;
      miClass = miUINT16;
      bytesPerElement = 2;
      break;
   case CV_16S:
      mxClass = mxINT16_CLASS;
      miClass = miINT16;
      bytesPerElement = 2;
      break;
   case CV_32S:
      mxClass = mxINT32_CLASS;
      miClass = miINT32;
      bytesPerElement = 4;
      break;
   case CV_32F:
      mxClass = mxSINGLE_CLASS;
      miClass = miSINGLE;
      bytesPerElement = 4;
      break;
   case CV_64F:
      mxClass = mxDOUBLE_CLASS;
      miClass = miDOUBLE;
      bytesPerElement = 8;
      break;
   default:
      return;
   }

   //==================Mat-file header (128 bytes, page 1-5)==================
   text = new char[textLen]; // Human-readable text.
   memset( text, ' ', textLen );
   text[textLen-1] = '\0';
   const char* t = "MATLAB 5.0 MAT-file, Platform: PCWIN";
   memcpy( text, t, strlen(t) );

   subsysOffset = new char[subsysOffsetLen]; // Zeros for us.
   memset( subsysOffset, 0x00, subsysOffsetLen );
   ver = new char[verLen];
   ver[0] = 0x00;
   ver[1] = 0x01;

   fwrite( text, 1, textLen, fp );
   fwrite( subsysOffset, 1, subsysOffsetLen, fp );
   fwrite( ver, 1, verLen, fp );
   // Endian indicator. MI will show up as "MI" on big-endian
   // systems and "IM" on little-endian systems.
   fwrite( &MI, 2, 1, fp );
   //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   //===================Data element tag (8 bytes, page 1-8)==================
   bytes = 16 + 24 + (8 + strlen(var_name) + (8-(strlen(var_name)%8))%8)
      + (8 + rows*cols*chans*bytesPerElement);
   fwrite( &miMATRIX, 4, 1, fp ); // Data type.
   fwrite( &bytes, 4, 1, fp); // Data size in bytes.
   //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   //====================Array flags (16 bytes, page 1-15)====================
   bytes = 8;
   fwrite( &miUINT32, 4, 1, fp );
   fwrite( &bytes, 4, 1, fp );
   flags = 0x00; // Complex, logical, and global flags all off.

   tmp32 = 0;
   tmp32 = (flags << 8 ) | (mxClass);
   fwrite( &tmp32, 4, 1, fp );

   fwrite( &zero, 4, 1, fp ); // Padding to 64-bit boundary.
   //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   //===============Dimensions subelement (24 bytes, page 1-17)===============
   bytes = 12;
   fwrite( &miINT32, 4, 1, fp );
   fwrite( &bytes, 4, 1, fp );

   fwrite( &rows, 4, 1, fp );
   fwrite( &cols, 4, 1, fp );
   fwrite( &chans, 4, 1, fp );
   fwrite( &zero, 4, 1, fp ); // Padding to 64-bit boundary.
   //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   //==Array name (8 + strlen(varName) + (8-(strlen(varName)%8))%8 bytes, page 1-17)==
   bytes = strlen(var_name);

   fwrite( &miINT8, 4, 1, fp );
   fwrite( &bytes, 4, 1, fp );
   fwrite( var_name, 1, bytes, fp );

   // Pad to nearest 64-bit boundary.
   padBytes = (8-(bytes%8))%8;
   fwrite( &zero, 1, padBytes, fp );
   //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   //====Matrix data (rows*cols*chans*bytesPerElement+8 bytes, page 1-20)=====
   bytes = rows*cols*chans*bytesPerElement;
   fwrite( &miClass, 4, 1, fp );
   fwrite( &bytes, 4, 1, fp );

   for( k = 0; k < chans; ++k )
   {
      if( doBgrSwap )
      {
         k2 = (k==0)? 2 : ((k==2)? 0 : 1);
      }
      else
         k2 = k;

      for( j = 0; j < cols; ++j )
      {
         for( i = 0; i < rows; ++i )
         {
            rowPtr = mat.data + mat.step*i;
            fwrite( rowPtr + (chans*j + k2)*bytesPerElement, bytesPerElement, 1, fp );
         }
      }
   }

   // Pad to 64-bit boundary.
   padBytes = (8-(bytes%8))%8;
   fwrite( &zero, 1, padBytes, fp );
   //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   fclose(fp);
   delete[] text;
   delete[] subsysOffset;
   delete[] ver;
}

void Utilities::ResizeImages(int number_of_frames, string file_name_in, string file_name_out,cv::Size size){
	
	string name;
	char snum[5];
	cv::Mat img,aux;

	for(int i=1; i<= number_of_frames; i++){

		sprintf(snum, "%d", i);
		name =file_name_in + string(snum) + JPG;

		img = cv::imread(name,CV_LOAD_IMAGE_GRAYSCALE);

		if(img.empty()){
		    cerr << "Missing input file names" << endl;
            exit(-1);
		}

		cv::resize(img,aux,size);

		name = file_name_out;
		name += string(snum) + JPG;
		cv::imwrite(name,aux);
	}
}

void Utilities::WriteFlowToFiles(vector<cv::Mat> h_flow_u, vector<cv::Mat> h_flow_v, string base_name_u, string base_name_v){

	cv::FileStorage fs;
	char snum[5];

	string file;

	for(unsigned int i=0; i< h_flow_u.size();i++){

		sprintf(snum, "%d", i);

		file= base_name_u +  string(snum) + ".xml";
		fs.open(file,cv::FileStorage::WRITE);
		fs << "flujooptico" << h_flow_u[i];
		fs.release();

		file= base_name_v + string(snum) + ".xml";
		fs.open(file,cv::FileStorage::WRITE);
		fs << "flujooptico" << h_flow_v[i];
		fs.release();
	}
}

OpticalFlowManagement::OpticalFlowManagement(){
}

OpticalFlowManagement::OpticalFlowManagement(int rows, int cols, int frames)
{
	rows_ = rows;
	cols_= cols;
	frames_ = frames;
}

OpticalFlowManagement::OpticalFlowManagement(int frames){

	frames_ = frames;
	cols_ = rows_ = 0;
}

OpticalFlowManagement::~OpticalFlowManagement(void)
{
}

cv::gpu::GpuMat OpticalFlowManagement::LoadImagesFromCpuToGpuPinnedMemory(string base_name){

	char snum[5];
	int index = 0;
	string file;

	cv::Mat h_header;
	cv::Mat h_roi,img;

	cv::gpu::GpuMat d_images;
	
	//It is necessary if you want to upload the images using pinned memory
	cv::gpu::Stream stream;

	cv::gpu::CudaMem h_imagesCudaMem = cv::gpu::CudaMem(cv::Size(cols_,rows_*frames_),0);
	cout << h_imagesCudaMem.rows << " " << h_imagesCudaMem.cols << endl;
	h_header= h_imagesCudaMem;
	
	 for(int i=1; i<= frames_; i++){

		sprintf(snum, "%d", i);
		file= base_name +  string(snum) + JPG;

		//The loop goes throught the rows and not the cols
		h_roi = h_header(cv::Rect(0,index,cols_,rows_));
		index += rows_;
		
		img = cv::imread(file,CV_LOAD_IMAGE_GRAYSCALE);

		if(img.empty()){
		    cerr << "Missing input file names" << endl;
            exit(-1);
		}

		//Every image is loaded in its corresponding array place
		img.copyTo(h_roi);
	 }

	 //The upload to gpu memory is done when all the images have been loaded from the files
	 //rather than going one by one.
	 stream.enqueueUpload(h_imagesCudaMem,d_images);

	 //The program waits until the upload finishes
	 stream.waitForCompletion();

	 h_imagesCudaMem.release();

	 return d_images;
}

cv::gpu::GpuMat OpticalFlowManagement::LoadImagesFromCpuToGpuPageableMemory(string base_name){

	char snum[5];
	string file;
	int index = 0;

	cv::Mat h_roi,img;

	cv::gpu::GpuMat d_images;
	cv::Mat h_images; 

	h_images.create(cv::Size(cols_,rows_*frames_),0);

	 for(int i=1; i<= frames_; i++){

		sprintf(snum, "%d", i);
		file = base_name +  string(snum) + JPG;

		//The loop goes throught the rows and not the cols
		h_roi = h_images(cv::Rect(0,index,cols_,rows_));
		index += rows_;

		img = cv::imread(file,CV_LOAD_IMAGE_GRAYSCALE);

		if(img.empty()){
		    cerr << "Missing input file names" << endl;
            exit(-1);
		}

		//Every image is loaded in its corresponding array place
		img.copyTo(h_roi);
	 }

	  //The upload to gpu memory is done when all the images have been loaded from the files
	  d_images.upload(h_images);

	  h_images.release();

	  return d_images;
}

void OpticalFlowManagement::CalculateOpticalFlowGPUVector(const cv::gpu::GpuMat d_images, vector<cv::gpu::GpuMat> d_flow_u, vector<cv::gpu::GpuMat> d_flow_v){

	int index = 0;

	cv::gpu::GpuMat d_image_prev;
	cv::gpu::GpuMat d_image_next;

	cv::gpu::GpuMat d_flow_u_aux,d_flow_v_aux;

	// Farneback Optical Flow is used
	cv::gpu::FarnebackOpticalFlow d_flow;

	//Default options
	d_flow.numLevels = 3;
	d_flow.winSize = 9;


	//Main Loop
	 for(int i=0; i< d_images.rows - rows_; i+= rows_){

		 // Images are saved in d_images
		 // Two images are obtained to calculate the optical flow
		 d_image_prev = d_images(cv::Rect(0,i,cols_,rows_));
		 d_image_next = d_images(cv::Rect(0,i+rows_,cols_,rows_));

		 //A new GpuMat is added to d_flowU and d_flowV arrays
		 d_flow_u.push_back(cv::gpu::GpuMat());
		 d_flow_v.push_back(cv::gpu::GpuMat());

		 cout << "Flow "<<index<< "..."<<endl;

		 //Optical flow is calculated
		 d_flow(d_image_prev,d_image_next,d_flow_u_aux,d_flow_v_aux);

		 //Optical flow components are saved in d_flowU and d_flowV arrays
		d_flow_u_aux.copyTo(d_flow_u[index]);
		d_flow_v_aux.copyTo(d_flow_v[index]);
	
		 index++;	 
	 }

	 //Memory is released
	 d_image_prev.release();
	 d_image_next.release();
}

void OpticalFlowManagement::CalculateOpticalFlowGPUMat(const cv::gpu::GpuMat d_images, cv::gpu::GpuMat &d_flow_u, cv::gpu::GpuMat &d_flow_v){

	int index = 0;

	//Auxiliar memory
	cv::gpu::GpuMat d_image_prev;
	cv::gpu::GpuMat d_image_next;
	cv::gpu::GpuMat d_flow_u_aux,d_flow_v_aux;

	// Farneback Optical Flow is used
	cv::gpu::FarnebackOpticalFlow d_flow;

	//Default options
	d_flow.numLevels = 3;
	d_flow.winSize = 9;

	//Coninuous Memory is created to save both x and y components of the optical flow

	d_flow_u = cv::gpu::createContinuous(rows_*frames_,cols_,CV_32F);
	d_flow_v = cv::gpu::createContinuous(rows_*frames_,cols_,CV_32F);

	 //Main Loop
	 for(int i=0; i< d_images.rows- rows_; i+= rows_){

		 // Images are saved in d_images
		 // Two images are obtained to calculate the optical flow
		 d_image_prev = d_images(cv::Rect(0,i,cols_,rows_));
		 d_image_next = d_images(cv::Rect(0,i+rows_,cols_,rows_));

		 cout << "Flow "<<index<< "..."<<endl;
		 
		  //Optical flow components will be saved in d_completeFlowU and d_completeFlowV GpuMat's
		 d_flow_u_aux = d_flow_u(cv::Rect(0,i,cols_,rows_));
		 d_flow_v_aux = d_flow_v(cv::Rect(0,i,cols_,rows_));

		 //Optical flow is calculated
		 d_flow(d_image_prev,d_image_next,d_flow_u_aux,d_flow_v_aux);
 
		 index++;	 
	 }

	 //Memory is released
	 d_image_prev.release();
	 d_image_next.release();
}

void OpticalFlowManagement::ReadFlowFromFilesPageableMemory(string base_name_optical_flow_u, string base_name_optical_flow_v, cv::Mat &h_flow_u, cv::Mat &h_flow_v){

	cv::FileStorage fs;
	char snum[5];
	int index;
	cv::Rect rectaux;

	cv::Mat h_aux1,h_aux2;

	string file;

	h_flow_u = cv::Mat(rows_*(frames_-1),cols_,CV_32F);
	h_flow_v = cv::Mat(rows_*(frames_-1),cols_,CV_32F);

	index = 0;

	for(int i=0; i< h_flow_u.rows; i+= rows_){

		cout << "Loading flow "<<index<<endl;

		sprintf(snum, "%d", index);

		rectaux = cv::Rect(0,i,cols_,rows_);

		file= base_name_optical_flow_u +  string(snum) + ".xml";
		if(!fs.open(file,cv::FileStorage::READ)){
			cerr << "Error al abrir fichero xml u" << endl;
			exit(-1);
		}
		fs["flujooptico"] >> h_aux1;
		fs.release();
		
		file= base_name_optical_flow_v + string(snum) + ".xml";
		if(!fs.open(file,cv::FileStorage::READ)){
			cerr << "Error al abrir fichero xml v" << endl;
			exit(-1);
		}
		fs ["flujooptico"] >> h_aux2;
		fs.release();

		h_aux1.copyTo(h_flow_u(rectaux));
		h_aux2.copyTo(h_flow_v(rectaux));

		index++;
	}
}

void OpticalFlowManagement::ReadFlowFromFilesPinnedMemory(string base_name_optical_flow_u, string base_name_optical_flow_v, cv::gpu::CudaMem &h_flow_u, cv::gpu::CudaMem &h_flow_v){

	cv::FileStorage fs;
	char snum[5];
	int index;
	cv::Rect rectaux;

	cv::Mat h_header_u, h_header_v;
	cv::Mat h_aux1,h_aux2;

	string file;

    h_flow_u.create(rows_*(frames_-1),cols_,CV_32F);
	h_flow_v.create(rows_*(frames_-1),cols_,CV_32F);

	h_header_u = h_flow_u;
	h_header_v = h_flow_v;

	index = 0;

	for(int i=0; i< h_flow_v.rows; i+= rows_){

		cout << "Loading flow "<<index<<endl;

		sprintf(snum, "%d", index);

		rectaux = cv::Rect(0,i,cols_,rows_);

		file= base_name_optical_flow_u +  string(snum) + ".xml";
		fs.open(file,cv::FileStorage::READ);
		fs["flujooptico"] >> h_aux1;
		fs.release();

		file= base_name_optical_flow_v + string(snum) + ".xml";
		fs.open(file,cv::FileStorage::READ);
		fs ["flujooptico"] >> h_aux2;
		fs.release();

		h_aux1.copyTo(h_header_u(rectaux));
		h_aux2.copyTo(h_header_v(rectaux));

		index++;
	}
}

void OpticalFlowManagement::DownloadFlowMatPageableMemory(const cv::gpu::GpuMat d_flow_u, const cv::gpu::GpuMat d_flow_v, cv::Mat &h_flow_u, cv::Mat &h_flow_v){

	//Optical flow components are downloaded to pageable memory
	d_flow_u.download(h_flow_u);
	d_flow_v.download(h_flow_v);
}

void OpticalFlowManagement::DownloadFlowMatPinnedMemory(const cv::gpu::GpuMat d_flow_u, const cv::gpu::GpuMat d_flow_v, cv::gpu::CudaMem &h_flow_u, cv::gpu::CudaMem &h_flow_v){

	cv::Mat header_h_flow_u;
	cv::Mat header_h_flow_v;

	//Pinned memory is allocated
	h_flow_u.create(cv::Size(d_flow_u.cols,d_flow_u.rows),d_flow_u.type());
	h_flow_v.create(cv::Size(d_flow_v.cols,d_flow_v.rows),d_flow_v.type());

	header_h_flow_u = h_flow_u.createMatHeader();
	header_h_flow_v = h_flow_v.createMatHeader();

	//Optical Flow is downloaded
	 d_flow_u.download(header_h_flow_u);
	 d_flow_v.download(header_h_flow_v);
}

void OpticalFlowManagement::DownloadFlowVectorPageableMemory(vector<cv::gpu::GpuMat> d_flow_u, vector<cv::gpu::GpuMat> d_flow_v, vector<cv::Mat> h_flow_u, vector<cv::Mat> h_flow_v){

  for(int i=0;i< frames_-1; i++){

	 //Pageable memory is allocated
	 h_flow_u.push_back(cv::Mat());
	 h_flow_v.push_back(cv::Mat());

	 //Optical flow is downloaded
	 d_flow_u[i].download(h_flow_u[i]);
	 d_flow_v[i].download(h_flow_v[i]);
  }
}

void OpticalFlowManagement::DownloadFlowVectorPinnedMemory(vector<cv::gpu::GpuMat> d_flow_u, vector<cv::gpu::GpuMat> d_flow_v, vector<cv::gpu::CudaMem> h_flow_u, vector<cv::gpu::CudaMem> h_flow_v){

	cv::Mat header_h_flow_u;
	cv::Mat header_h_flow_v;

	 for(int i=0;i< frames_-1; i++){

	 //Pageable memory is allocated
	 h_flow_u.push_back(cv::gpu::CudaMem(d_flow_u[i].rows,d_flow_u[i].cols,d_flow_u[i].type()));
	 h_flow_v.push_back(cv::gpu::CudaMem(d_flow_v[i].rows,d_flow_v[i].cols,d_flow_v[i].type()));

	header_h_flow_u = h_flow_u[i].createMatHeader();
	header_h_flow_v = h_flow_v[i].createMatHeader();

	 //The optical flow is downloaded
	 d_flow_u[i].download(header_h_flow_u);
	 d_flow_v[i].download(header_h_flow_v);
  }
}




Times::Times(void)
{
	cartesians_to_polars_timer_ = 0;
	comparisons_timer_ = 0;
	histogram_timer_ = 0;
	upload_weights_timer_ = 0;
	transpose_timer_ = 0;
	accumulate_timer_ = 0;
	normalize_timer_ = 0;
	program_timer_ = 0;
	hof_timer_ = 0;
	pyramidal_hof_timer_ = 0;
	weights_timer_ = 0;
	reserve_memory_timer_ = 0;
}

void Times::ResetTimers(){

	cartesians_to_polars_timer_ = 0;
	comparisons_timer_ = 0;
	histogram_timer_ = 0;
	upload_weights_timer_ = 0;
	transpose_timer_ = 0;
	accumulate_timer_ = 0;
	normalize_timer_ = 0;
	program_timer_ = 0;
	hof_timer_ = 0;
	pyramidal_hof_timer_ = 0;
	weights_timer_ = 0;
	reserve_memory_timer_ = 0;
}

Times::~Times(void)
{
}

void Times::StartTime(){

#if defined _WIN32
	start_time_ = clock();
#else
	gettimeofday(&start_time_,NULL);
#endif
}

void Times::StopTime(){

#if defined _WIN32
	finish_time_ = clock();
#else
	gettimeofday(&finish_time_,NULL);
#endif
}

double Times::GetTimeV1(){

	double result = 0;

#if defined _WIN32
	result = (double)(finish_time_ - start_time_)/CLOCKS_PER_SEC;
#else
	result += (finish_time_.tv_sec - start_time_.tv_sec) * 1000.0;
	result += (finish_time_.tv_usec - start_time_.tv_usec) / 1000.0;
#endif

	return result;
}

double Times::GetTimeV2(){

	double result = 0;

#if defined _WIN32
	result = difftime(finish_time_, start_time_);
#else
	result += (finish_time_.tv_sec - start_time_.tv_sec) * 1000.0;
	result += (finish_time_.tv_usec - start_time_.tv_usec) / 1000.0;
#endif

	return result;
}

void Times::PrintCartToPolarTimer(){

	cout << "----------------------------------------------------------"<<endl;
	cout << "\t Calculate Cartesians to Polars: " << cartesians_to_polars_timer_<<" ms"<<endl;
	cout << "----------------------------------------------------------"<<endl;
}

void Times::PrintPyramidalTimer(){

	cout << "----------------------------------------------------------"<<endl;
	cout << "\t Calculate Pyramidal hof: " << pyramidal_hof_timer_<<" ms"<<endl;
	cout << "----------------------------------------------------------"<<endl;

}

void Times::PrintHofTimer(){

	cout << "----------------------------------------------------------"<<endl;
	cout << "\t Total: " << hof_timer_<<" ms"<<endl;
	cout << "----------------------------------------------------------"<<endl;
}

void Times::PrintTimesHof(){

	cout << "----------------------------------------------------------"<<endl;
	cout << "\t Calculate Weights: " << weights_timer_<<" ms"<<endl;
	cout << "----------------------------------------------------------"<<endl;
	cout << " \t Histograms:" << endl;
	cout << "----------------------------------------------------------"<<endl;
	cout << "\t Calculate comparisons: " << comparisons_timer_<<" ms"<<endl;
	cout << "\t Calculate dot products: " << histogram_timer_<<" ms"<<endl;
	cout << "----------------------------------------------------------"<<endl;
}

void Times::PrintTimesHofGpu(){

	cout << "----------------------------------------------------------"<<endl;
	cout << "\t Reserve memory: " << reserve_memory_timer_<<" ms"<<endl;
	cout << "\t Calculate Weights: " << weights_timer_<<" ms"<<endl;
	cout << "\t Upload Weights: " << upload_weights_timer_ << " ms"<<endl;
	cout << "----------------------------------------------------------"<<endl;
	cout << " \t Histograms:" << endl;
	cout << "----------------------------------------------------------"<<endl;
	cout << "\t Calculate comparisons: " << comparisons_timer_<<" ms"<<endl;
	cout << "\t Calculate dot products: " << histogram_timer_<<" ms"<<endl;
	cout << "----------------------------------------------------------"<<endl;

}

void Times::PrintTimesAcumulatedHof(){

	cout << "----------------------------------------------------------"<<endl;
	cout << "\t Calculate accumulated histograms: " << accumulate_timer_<<" ms"<<endl;
	cout << "\t Normalize accumulated histograms: " << normalize_timer_<<" ms"<<endl;
	cout << "----------------------------------------------------------"<<endl;
}

void Times::PrintProgramTimer(){

	cout << "----------------------------------------------------------"<<endl;
	cout << "\t TOTAL: " << program_timer_<<" ms"<<endl;
	cout << "----------------------------------------------------------"<<endl;

}


SVMPahof::SVMPahof(void){

}

SVMPahof::SVMPahof(string srcXml){
	
	//Data train is loaded
	svm_.load(srcXml.c_str());
}

SVMPahof::~SVMPahof(void){


}

void SVMPahof::TrainSVM(cv::Mat data, int number_of_clases, int size_class, string src_xml){

	  // Data for visual representation
    cv::Mat labels   (data.rows, 1, CV_32FC1);
	 int index = 0;

    //------------------------- Set up the labels for the classes ---------------------------------
	for(int i=0; i< number_of_clases; i++){

		labels.rowRange(index, index + size_class).setTo(i);  
		index += size_class; 
	}

    //------------------------ 2. Set up the support vector machines parameters --------------------
    CvSVMParams params;
    params.svm_type    = cv::SVM::C_SVC;
    params.C           = 0.1;
    params.kernel_type = cv::SVM::LINEAR;
    params.term_crit   = cv::TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);

    //------------------------ 3. Train the svm ----------------------------------------------------
    cout << "Starting training process" << endl;
    CvSVM svm;
   
	svm.train(data, labels, cv::Mat(), cv::Mat(), params);
    cout << "Finished training process" << endl;

	svm.save(src_xml.c_str());
}

void SVMPahof::ClassifySVM(cv::Mat data, string src_xml){

	float response;

	cout << "Starting classify process" << endl;
    CvSVM svm;

	//Data train is loaded
	svm.load(src_xml.c_str());

	for(int i=0; i<data.rows; i++){

		//Every row is an element that must be classified
		response = svm.predict(data.row(i));

		switch ((int)response)
		{
		case 0:
			cout << "Person walking detected" << endl;
			break;
		case 1:
			cout << "Person jogging detected" << endl;
			break;
		case 2:
			cout << "Person boxing detected" << endl;
			break;
		case 3:
			cout << "Person hand clapping detected" << endl;
			break;
		case 4:
			cout << "Person hand waving detected" << endl;
			break;
		default:
			cout << "No motion detected" << endl; 
			break;
		}
	}

    cout << "Finished classify process" << endl;
}

string SVMPahof::PredictSVM(cv::Mat data){
	
	float response;
	string out;
	 response = svm_.predict(data);

	 
		switch ((int)response)
		{
		case 0:
			cout << "Person walking detected" << endl;
			out = "Person walking detected";
			break;
		case 1:
			cout << "Person jogging detected" << endl;
			out = "Person jogging detected";
			break;
		case 2:
			cout << "Person boxing detected" << endl;
			out = "Person boxing detected";
			break;
		case 3:
			cout << "Person hand clapping detected" << endl;
			out = "Person hand clapping detected";
			break;
		case 4:
			cout << "Person hand waving detected" << endl;
			out = "Person hand waving detected";
			break;
		default:
			cout << "No motion detected" << endl; 
			out = "No motion detected";
			break;
		}

		return out;
}


Args::Args()
{
    make_gray_ = false;
    resize_src_ = false;
    width_ = 640;
    height_ = 480;
    scale_ = 1.05;
    nlevels_ = 13;
    gr_threshold_ = 8;
    hit_threshold_ = 1.4;
    hit_threshold_auto_ = false;
    win_width_ = 48;
    win_stride_width_ = 8;
    win_stride_height_ = 8;
    gamma_corr_ = true;

	video_ = false;
	images_ = false;
	xmls_ = false;

	frames_ = 0;

	gpu_ = false;
	train_ = false;
	classify_ = false;
	xml_ = false;

	len_ = 20;
	step_ = 2;
	n_ang_ = 8;

    fps_ = false;

	rows_ = ROWS;
	cols_ = COLS;
}

Args Args::Read(int argc, char** argv)
{
    Args args;

	if(argc == 1){
		PrintHelp();
		cout << "Press to end...";
		getchar();
		exit(EXIT_FAILURE);

	}
	else{
		if(string(argv[1]) == "--video"){
			if(argc < 4){
				PrintHelp();
				cout << "Press to end...";
				getchar();
				exit(EXIT_FAILURE);
			}
          args.video_ = true;
		  args.src_ = argv[2];
		  CheckInteger(argv[3]);
		  args.frames_ =  atoi(argv[3]);
		}
		else{
			if(string(argv[1]) == "--images"){
			 if(argc < 4){
				PrintHelp();
				cout << "Press to end...";
				getchar();
				exit(EXIT_FAILURE);
			 }
			 args.images_ = true;
		     args.src_ = argv[2];	
			 CheckInteger(argv[3]);
		     args.frames_ =  atoi(argv[3]);
			}
			else{		
			if(string(argv[1]) == "--xmls"){
			 if(argc < 4){
				PrintHelp();
				cout << "Press to end...";
				getchar();
				exit(EXIT_FAILURE);
			 }
			 args.xmls_ = true;
		     args.src_ = argv[2];
			 CheckInteger(argv[3]);
		     args.frames_ =  atoi(argv[3]);
			}
			}
		}
	}
	

    for (int i = 4; i < argc; i++){

		if (string(argv[i]) == "--gpu") args.gpu_ = true;
		else if (string(argv[i]) == "--xml") args.xml_ = true;
		else if (string(argv[i]) == "--fps") args.fps_ = true;
		else if (string(argv[i]) == "--train")    {args.train_ = true;    CheckErrors(argc,i); CheckXML(argv[i+1],cv::FileStorage::WRITE); args.src_svm_ = argv[i+1];}
		else if (string(argv[i]) == "--classify") {args.classify_ = true; CheckErrors(argc,i); CheckXML(argv[i+1],cv::FileStorage::READ); args.src_svm_ = argv[i+1];}
		else if (string(argv[i]) == "--step") {CheckErrors(argc,i); CheckInteger(argv[i+1]);   args.step_ = atoi(argv[i+1]);}
		else if (string(argv[i]) == "--len") {CheckErrors(argc,i);  CheckInteger(argv[i+1]);   args.len_ =  atoi(argv[i+1]);}
		else if (string(argv[i]) == "--nang") {CheckErrors(argc,i); CheckInteger(argv[i+1]);   args.n_ang_ = atoi(argv[i+1]);}
		else if (string(argv[i]) == "--nlevels") {CheckErrors(argc,i); CheckInteger(argv[i+1]);args.nlevels_ = atoi(argv[i+1]);}
		else if (string(argv[i]) == "--gr_threshold") {CheckErrors(argc,i); CheckInteger(argv[i+1]); args.gr_threshold_ = atoi(argv[i+1]);}
		else if (string(argv[i]) == "--scale") {CheckErrors(argc,i);  CheckDouble(argv[i+1]); args.scale_ = atof(argv[i+1]);}			
		else if (string(argv[i]) == "--hit_threshold") {CheckErrors(argc,i); CheckDouble(argv[i+1]); args.hit_threshold_ = atof(argv[i+1]); args.hit_threshold_auto_ = false;}
		else if (string(argv[i]) == "--help") PrintHelp();	
    }
    return args;
}

void Args::PrintHelp(){

	    cout << "cuPaHOF algorithm.\n"
         << "\n Required (1):\n"
         << "  [--video <videoname> <frames>] # Frames source + total number of frames\n"
		 << "  [--images <base name> <images>] # Images source to calculate OF + total number of images\n"
		 << "  [--xmls  <base name> <of files>] # Files source with OF precalculated + number of OF files\n"
		 << "\n PaHof parameters:\n"
		 << "  [--step <step>] # Parameter step\n"
		 << "  [--len <len>] # Parameter len\n"
		 << "  [--nang <nang>] # Parameter nAng\n"
		 << "\n Detection parameters:\n"
		 << "  [--hit_threshold <double>] # classifying plane distance threshold (0.0 usually)\n"
		 << "  [--gr_threshold <int>] # merging similar rects constant\n"
         << "  [--scale <double>] # HOG window scale factor\n"
         << "  [--nlevels <int>] # max number of HOG window scales\n"
		 << "\n Visual parameters:\n"
		  << "  [--fps] # Show fps\n"
		 << "\n Others parameters:\n"
		 << "  [--gpu] # Use CUDA\n"
	     << "  [--classify <xmlfile>] # SVM Classify\n"
		 << "  [--xml] # Save data in XML\n"; 
}

void Args::CheckErrors(int argc, int i){
	
	if(argc == i+1){
		cerr<<"Error in parameters"<<endl;
		PrintHelp();
		cout << "Press to end...";
		getchar();
		exit(EXIT_FAILURE);
	}
}

void Args::CheckDouble(char *src){
		
	std::istringstream ss(src);
	double d;

	if (!(( ss >> d) && (ss >> std::ws).eof())){
		cerr<<"Error in parameters"<<endl;
		PrintHelp();
		cout << "Press to end...";
		getchar();
		exit(EXIT_FAILURE);
	}
}

void Args::CheckInteger(char * src){
		
	for(int i=0; i< (signed)strlen(src); i++){

		if(!isdigit(src[i])){
			cerr<<"Error in parameters"<<endl;
			PrintHelp();
			cout << "Press to end...";
			getchar();
			exit(EXIT_FAILURE);
		}
	}
}

void Args::CheckXML(char *src, int mode){

	cv::FileStorage fs;
	
	if(!fs.open(src,mode)){
		cerr << "Fatal error. Check your xml file for classfying" << endl << "Press to exit...";
		getchar();
		exit(EXIT_FAILURE);
	}	
	fs.release();
}


HumanMotionDetection::HumanMotionDetection(const Args& s)
{
    args_ = s;
	detections_ = 0;

    make_gray_ = args_.make_gray_;
    scale_ = args_.scale_;
	nlevels_ = args_.nlevels_;

    gr_threshold_ = args_.gr_threshold_;


    if (args_.hit_threshold_auto_)
        args_.hit_threshold_ = args_.win_width_ == 48 ? 1.4 : 0.;
  
	hit_threshold_ = args_.hit_threshold_;


    gamma_corr_ = args_.gamma_corr_;

    if (args_.win_width_ != 64 && args_.win_width_ != 48)
        args_.win_width_ = 64;

}

int HumanMotionDetection::Run()
{

	int counterFrames = 0;
	cv::Point point1, point2;
    cv::VideoWriter video_writer;
	cv::gpu::FarnebackOpticalFlow d_flow;
	cv::gpu::Stream stream;

	cv::Size size(args_.cols_,args_.rows_);
    cv::VideoCapture vc;
    cv::Mat h_frame_gray,h_frame_gray_2;
	bool changeToGray;
	bool firstFramePerson = false;
	vector<float> detector;

	cv::Mat h_frame[2];
	cv::Mat h_flow_u,h_flow_v;
	cv::gpu::GpuMat d_frame[2];
	cv::gpu::GpuMat d_frame_resized[2];
	vector<cv::Rect> found[2];

	cv::gpu::GpuMat d_flow_u,d_flow_v;
	cv::gpu::GpuMat d_magnitude,d_angle;
	cv::Mat h_magnitude,h_angle;
	

    cv::Size win_size(args_.win_width_, args_.win_width_ * 2); //(64, 128) or (48, 96)
    cv::Size win_stride(args_.win_stride_width_, args_.win_stride_height_);

	cv::namedWindow("Video", CV_WINDOW_NORMAL);

    // Create HOG descriptors and detectors here

    if (win_size == cv::Size(64, 128))
        detector = cv::gpu::HOGDescriptor::getPeopleDetector64x128();
    else
        detector = cv::gpu::HOGDescriptor::getPeopleDetector48x96();

    cv::gpu::HOGDescriptor gpu_hog(win_size, cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9,
                                   cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr_,
                                   cv::gpu::HOGDescriptor::DEFAULT_NLEVELS);

    gpu_hog.setSVMDetector(detector);

    if(!vc.open(args_.src_.c_str())){
		cerr << "can't open video file: " + args_.src_;
	    exit(EXIT_FAILURE);
	}
           
	gpu_hog.nlevels = nlevels_;

	vc >> h_frame[0];

	cout <<"Processing video..." << endl;
	//Until we detect a frame with someone
	while(!firstFramePerson && !h_frame[0].empty() && counterFrames < args_.frames_){
		
		WorkBegin();

		if(h_frame[0].channels() == 2)
			changeToGray = false;
		else
			changeToGray = true;

		if(!h_frame[0].empty()){
		
			if(changeToGray)
			    cvtColor(h_frame[0], h_frame_gray, CV_BGR2GRAY);
			else
				h_frame_gray = h_frame[0];

			d_frame[0].upload(h_frame_gray);
			gpu_hog.detectMultiScale(d_frame[0], found[0], hit_threshold_, win_stride,
                                     cv::Size(0, 0), scale_, gr_threshold_);

			if(found[0].size() > 0)
				firstFramePerson = true;
		}

		if(!firstFramePerson)
		 vc >> h_frame[0];

		if(args_.fps_)
		 cv::putText(h_frame[0], "FPS: " + WorkFps(), cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 100, 0), 2);

		if(!h_frame[0].empty()){
		 cv::imshow("Video", h_frame[0]);
		 cvWaitKey(1);

		counterFrames++;
		}
		 WorkEnd();
	}

	if(firstFramePerson)
	 detections_++;


	if(vc.isOpened()){
	vc >> h_frame[1];
	counterFrames++;


	cv::Rect r,r2;
	r =  found[0][0];
	cv::Mat img_to_show;
	changeToGray = true;

	while(!h_frame[1].empty() && counterFrames < args_.frames_){

		WorkBegin();

		img_to_show = h_frame[1];

		if(changeToGray)
			cvtColor(h_frame[1], h_frame_gray_2, CV_BGR2GRAY);
		else
			h_frame_gray_2 = h_frame[1];

		d_frame[1].upload(h_frame_gray_2);
		gpu_hog.detectMultiScale(d_frame[1], found[1], hit_threshold_, win_stride,
                                     cv::Size(0, 0), scale_, gr_threshold_);

		if(found[1].size() >0){


				detections_++;
				r2 = found[1][0];

				//Calculate the rect where both detections are included
				point1.x = cv::min(r2.tl().x,r.tl().x);
				point1.y = cv::min(r2.tl().y,r.tl().y);

				point2.x = cv::max(r2.br().x,r.br().x);
				point2.y = cv::max(r2.br().y,r.br().y);
					
				cv::Rect rect(point1,point2);	

				rectangle(img_to_show,rect.tl(), rect.br(), CV_RGB(0, 255, 0), 3);	

				//We resize both images to a given size (64x64 for example) and, at the same time, we use our cut area in both frames
				cv::gpu::resize(d_frame[0](rect),d_frame_resized[0],size);	
				cv::gpu::resize(d_frame[1](rect),d_frame_resized[1],size);	

				//Optical flow
				d_flow(d_frame_resized[0],d_frame_resized[1],d_flow_u,d_flow_v);

				 
				//Download results to the CPU
				d_flow_u.download(h_flow_u);
				d_flow_v.download(h_flow_v);

				d_flow_u.release();
				d_flow_v.release();


				//Accumulate Polars in a single Mat
			    if(detections_ == 2){

					h_flow_u.copyTo(h_flow_u_);
					h_flow_v.copyTo(h_flow_v_);
				}
				else{
					vconcat(h_flow_u_,h_flow_u,h_flow_u_);
					vconcat(h_flow_v_,h_flow_v,h_flow_v_);
				}

				r = r2;
				d_frame[1].copyTo(d_frame[0]);
		}

		if(args_.fps_)
		 cv::putText(img_to_show, "FPS: " + WorkFps(), cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 100, 0), 2);

		cv::imshow("Video", img_to_show);
		cvWaitKey(1);



		vc >> h_frame[1];
		counterFrames++;

		 WorkEnd();
	}
   }


	vc.release();
	cv::destroyWindow("Video");
	cout << "Number of detections: " << detections_ << endl;

	return 0;
}

cv::Mat HumanMotionDetection::GetOpticalFlowU(){
	return h_flow_u_;
}

cv::Mat HumanMotionDetection::GetOpticalFlowV(){
	return h_flow_v_;
}	

int HumanMotionDetection::GetDetections(){
	return detections_;
}

void HumanMotionDetection::WorkBegin(){ 
	work_begin_ = cv::getTickCount(); 
}

void HumanMotionDetection::WorkEnd()
{
    int64 delta = cv::getTickCount() - work_begin_;
    double freq = cv::getTickFrequency();
    work_fps_ = freq / delta;
}

string HumanMotionDetection::WorkFps() const
{
    stringstream ss;
    ss << work_fps_;
    return ss.str();
}