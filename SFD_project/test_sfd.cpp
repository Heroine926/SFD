#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <time.h>
#include <iosfwd>
#include <utility>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


#include "./soSFD.h"

using namespace std;
using namespace cv;

#define LIB_SFD "./sfd_FaceDetect.so"



extern "C"
{
	typedef int (*F_sfd_face_detection)(const void *p_cvmat, void *p_vec_vec_rect, int batchSize);
	typedef int (*F_sfd_face_detection_init)(const string& model_def, const string& trained_file, int gpu_id);
}

  int OffsetJ[4] = { 4, -4, -4, 4 };
    int OffsetI[4] = { -4, -4, 4, 4 };

    int sumB,sumG,sumR;
    for(int j = 0;j<srcImage.rows;j++)
    {
	for(int i = 0;i<srcImage.cols;i++)
	{
	    sumB = 0;sumG = 0;sumR = 0;
	    for(int k = 0;k<4;k++)
	    {
		int JJ = j + OffsetJ[k];
		int II = i + OffsetI[k];

		if(JJ < 0)
		{
		    JJ = 0;
		}
		else if(JJ >= srcImage.rows)
		{
		    JJ = srcImage.rows - 1;
		}
		if(II < 0)
		{
		    II = 0;
		}else if(II >= srcImage.cols)
		{
		    II = srcImage.cols - 1;
		}

		sumB += srcImage.at<Vec3b>(JJ,II)[0];
		sumG += srcImage.at<Vec3b>(JJ,II)[1];
		sumR += srcImage.at<Vec3b>(JJ,II)[2];
	    }

	    srcImage.at<Vec3b>(j,i)[2] = (sumR+2)>>2;
	    srcImage.at<Vec3b>(j,i)[1] = (sumG+2)>>2;
	    srcImage.at<Vec3b>(j,i)[0] = (sumB+2)>>2;
	}
    }
}


int main()
{
   
	string model_file  = "../caffe/models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt";
	
	string trained_file = "../caffe/models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel";
	string test_image_path = "./images/";
	string image_list_file = "./image_list.txt";
        
	int res = 0;
	int gpu_id = 2;

	void *handle = dlopen(LIB_SFD,RTLD_LAZY);
	if(!handle)
	{
		printf("%s\n",dlerror());
		exit(EXIT_FAILURE);
	}

	char *error;
	dlerror();

	F_sfd_face_detection sfd_face_detection = (F_sfd_face_detection)dlsym(handle,"sfd_face_detection");
	F_sfd_face_detection_init sfd_face_detection_init = (F_sfd_face_detection_init)dlsym(handle,"sfd_face_detection_init");
	if((error = dlerror()) != NULL)
	{
		printf("%s\n",error);
		exit(EXIT_FAILURE);
	}
	
	res = sfd_face_detection_init(model_file, trained_file,gpu_id);
	printf("=====================finish loading network!=============================\n\n\n");
	ifstream img_list;
	img_list.open(image_list_file.data());

	if (!img_list)
	{
		std::cerr << image_list_file << " open error." <<endl;;
		exit(1);
	}
	string img_name;
	clock_t start_time = clock();
	int counter = 0;

   
	int totalBoxNum = 0;
	FILE * bbx_file = fopen("/data/home/larainelu/DataSet/widerface/SFD_original_val.txt","w");
	
	if(bbx_file == NULL ) 
	    cout<<"file error"<<endl;
	

	
	
	bool end = false;
	int batchSize = 1 ;
	FILE* result = fopen("/data/home/larainelu/SFD/tools/taxi_my.txt","w");
	if(result == NULL) printf("result file error");
	while(!end)
    	{
	    	vector<cv::Mat> img_vec;
		vector<string> Img_path_vec;

		
		while( (img_vec.size() < batchSize) && (!end) )
		{
		    	if(!getline(img_list,img_name))
			{
				end = true;
				batchSize = img_vec.size();
				if(batchSize == 0)
					break;
			}

			
			string Img_path = test_image_path + img_name;						
			cv::Mat img = cv::imread(Img_path,-1);
			if(img.empty())
			{
				cout<<"ERROR: "<<Img_path<<" is empty"<<endl;
				continue;
			}
			Img_path_vec.push_back(Img_path);
			img_vec.push_back(img);
		}
		//cout << img_vec.size()<<endl;
		
		std::vector<std::vector<cv::Rect> > vrects;
		
		res = sfd_face_detection(&img_vec[0], &vrects, batchSize);

		
		//cout<<"res = "<<res<<endl;
		if(res != 0)
		{
			if(res == -1) 
			    cout<<"ERROR: p_cvmat is nullpr\n";
			if(res == -3)
			    cout<<"ERROR: img is empty\n";
			if(res == -5)
			    cout<<"ERROR: p_vec_vec_tect is nullpr\n";
			if(res == -7)
			    cout<<"ERROR: no target in this image\n";
			continue;
		}
		for(int j = 0;j < batchSize;j++)
		{
			//cout<<Img_path_vec[j]<<endl;
			std::vector<cv::Rect> &rects = vrects[j];
			//cout<<"rect num:"<<rects.size()<<endl;
			for(int i = 0;i < rects.size();i++)
			{
				cout<<Img_path_vec[j]<<"    ";
				cout<< rects[i].x<<"    ";
				cout<< rects[i].y<<"    ";
				cout<< rects[i].width<<"    ";
				cout<< rects[i].height<<"    ";
				cout<< endl;

				
				string path = Img_path_vec[j];
				fprintf(result,"%s    ",path.c_str());
				fprintf(result,"%d    ",rects[i].x);
				fprintf(result,"%d    ",rects[i].y);
				fprintf(result,"%d    ",rects[i].width);
				fprintf(result,"%d    \n",rects[i].height);
			
			}
		}
		
		counter = counter+batchSize;
	}
	fclose(result);
	//=====================================================================================================*/
	




	clock_t end_time = clock();
        double mean_time = double((end_time-start_time))/CLOCKS_PER_SEC/(counter)*1000;
	printf("====================== elasped %f ms per image =========================\n",mean_time);

	printf("total box num: %d\n",totalBoxNum);

	dlclose(handle);
     

	return 0;
}

