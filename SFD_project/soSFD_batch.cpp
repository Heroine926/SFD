#include<fstream>
#include<iostream>
#include<stdio.h>
#include<caffe/caffe.hpp>
//#include"./soSFD.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <time.h>



using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;


class faceDetection{
public:
	faceDetection(const string& model_def, const string& trained_file, const string& mean_file, const string& mean_value);

	vector<vector<float> > detector(const vector<cv::Mat> batch_img);

private:
	void SetMean(const string& mean_file, const string& mean_value);

        void WrapInputLayer(std::vector<cv::Mat> *input_channels);

	void Preprocess(const vector<cv::Mat>& img_vec,
		std::vector<cv::Mat>* input_channels);

private:

	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
public:

	int batch_size_;
};

faceDetection::faceDetection(const string& model_def, const string& trained_file , const string& mean_file, const string& mean_value)
{
   	#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
    	#else
	Caffe::set_mode(Caffe::GPU);
    	#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_def, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
        SetMean(mean_file,mean_value);

	batch_size_ = 3;
}


std::vector<vector<float> > faceDetection::detector(const vector<cv::Mat> img_vec) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(batch_size_, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();
	
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);
        Preprocess(img_vec, &input_channels);
	
	cout<<"forward"<<endl;
	net_->Forward();

	Blob<float>* result_blob =  net_->output_blobs()[0];
        const float* result = result_blob->cpu_data();
        const int num_det = result_blob->height();

	cout<<"output height:"<<result_blob->height()<<endl;
	cout<<"output width:"<<result_blob->width()<<endl;
	cout<<"output nums:"<<result_blob->num()<<endl;
	cout<<"output channels:"<<result_blob->channels()<<endl;

        vector<vector<float> > detections;
        for (int k = 0; k < num_det; ++k)
	{
                if (result[0] == -1) 
		{
                        // Skip invalid detection.
                        result += 7;
                        continue;
                }
                vector<float> detection(result, result + 7);
                detections.push_back(detection);
                result += 7;
        }
        return detections;
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void faceDetection::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();

	//cout<<"input_layer->num()"<<input_layer->num()<<endl;
	cout<<"width: "<<width<<endl;
	cout<<"height: "<<height<<endl;
	for (int i = 0; i < input_layer->channels() * input_layer->num(); ++i)
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void faceDetection::Preprocess(const vector<cv::Mat>& img_vec, std::vector<cv::Mat>* input_channels)
{
	/* Convert the input image to the input image format of the network. */
    vector<cv::Mat>::const_iterator b = input_channels->begin();

    for(int i=0; i<img_vec.size(); i++)
    {
	cv::Mat sample;
        cv::Mat img = img_vec[i];
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;
	
	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;
	
	cv::Mat sample_float;
	if (num_channels_ == 3)
	    sample_resized.convertTo(sample_float, CV_32FC3);
	else
	    sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	sample_normalized = sample_float;
	cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	std::vector<cv::Mat> vec_tmp(b + i*3, b + i*3 + 3);
	cv::split(sample_normalized, vec_tmp);
    }
    //cout<< input_channels->at(0).data<<endl;

    /*for(int i=0; i<input_channels->size(); i++)
    {
	cout<<"channel size"<<input_channels->size()<<endl;
	for(int j=0;j < 10;j++)
	{
	    cout << input_channels[i].at<int>(0,j)<<"    ";
	}
	cout<<endl;
    }*/
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
	              == net_->input_blobs()[0]->cpu_data())
	          << "Input channels are not wrapping the input layer of the network.";

}


/* Load the mean file in binaryproto format. */
void faceDetection::SetMean(const string& mean_file, const string& mean_value) {
	cv::Scalar channel_mean;
	if (!mean_file.empty()) {
		CHECK(mean_value.empty()) <<
			"Cannot specify mean_file and mean_value at the same time";
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

		/* Convert from BlobProto to Blob<float> */
		Blob<float> mean_blob;
		mean_blob.FromProto(blob_proto);
		CHECK_EQ(mean_blob.channels(), num_channels_)
			<< "Number of channels of mean file doesn't match input layer.";

		/* The format of the mean file is planar 32-bit float BGR or grayscale. */
		std::vector<cv::Mat> channels;
		float* data = mean_blob.mutable_cpu_data();
		for (int i = 0; i < num_channels_; ++i) {
			/* Extract an individual channel. */
			cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
			channels.push_back(channel);
			data += mean_blob.height() * mean_blob.width();
		}

		/* Merge the separate channels into a single image. */
		cv::Mat mean;
		cv::merge(channels, mean);

		/* Compute the global mean pixel value and create a mean image
		 * filled with this value. */
		channel_mean = cv::mean(mean);
		mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
	}
	if (!mean_value.empty()) {
		CHECK(mean_file.empty()) <<
			"Cannot specify mean_file and mean_value at the same time";
		stringstream ss(mean_value);
		vector<float> values;
		string item;
		while (getline(ss, item, ',')) {
			float value = std::atof(item.c_str());
			values.push_back(value);
		}
		CHECK(values.size() == 1 || values.size() == num_channels_) <<
			"Specify either 1 mean_value or as many as channels: " << num_channels_;

		std::vector<cv::Mat> channels;
		for (int i = 0; i < num_channels_; ++i) {
			/* Extract an individual channel. */
			cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
					cv::Scalar(values[i]));
			channels.push_back(channel);
		}
		cv::merge(channels, mean_);
	}
}




//===================================================================
int main()
{
	/*	
        if (argc != 8) 
	{
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< "test_image_path image_list.txt "
			<< "result_dets.txt output_path" << std::endl;
		return 1;
	}

	::google::InitGoogleLogging(argv[0]);
	
	string model_file = argv[1];
	string trained_file = argv[2];
	string mean_file    = argv[3];
	string mean_value   = argv[4];
	string test_image_path = argv[5];
	string image_list_file = argv[6];
	char* result_file = argv[7];
	string output_path = argv[8];
	*/

	string model_file = "/data/home/larainelu/SFD/caffe/models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt";
	string trained_file = "/data/home/larainelu/SFD/caffe/models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel";
	string mean_file = "";
	string mean_value = "104,117,123";
	string test_image_path = "/data/home/larainelu/DataSet/h265Img/";
	string image_list_file = "/data/home/larainelu/DataSet/h265Img/image_list.txt";
	char result_file[100] = "/data/home/larainelu/DataSet/h265Img/detect_res_test.txt";
	//string output_path = "/data/home/larainelu/SFD/output/";
        
        printf("==========================\ninisializing network\n");
	faceDetection faceDetection(model_file, trained_file, mean_file, mean_value);
        printf("finish load network!\n===============================\n");

	ifstream img_list;
	img_list.open(image_list_file.data());

	if (!img_list)
	{
		std::cerr << image_list_file << " open error." <<endl;;
		exit(1);
	}

        FILE* res_file = fopen(result_file, "w");
        if(!res_file)
        {
		std::cerr << result_file << " open| error." << endl;
		exit(1);
	}

	int count = 1;
	string img_name;
	clock_t start_time = clock();
	//detections.clear();
	while(getline(img_list, img_name))
	{
		printf("the %dth picture...\n",count);
		string read_ext = ".jpg";
         	string Img_path = test_image_path + img_name + read_ext;
		
		cv::Mat img = cv::imread(Img_path,-1);
		CHECK(!img.empty()) << "Unable to decode image " << Img_path;
		int width = img.cols;
		int height = img.rows;
		
		vector<cv::Mat> batch_img;
		for(int i = 0;i < faceDetection.batch_size_;i++)
		{
		    batch_img.push_back(img);
		}
		vector<vector<float> > detections = faceDetection.detector(batch_img);
		

		/*for (size_t i = 0; i < detections.size(); i++)
		{
			if (detections[i][2] < 0.5)
			{
				detections.erase(detections.begin() + i);
			}
		}*/
		printf("there are %d boxes in this picture\n",int(detections.size()));
		//fprintf(res_file, "%s\n", img_name.c_str());
		//fprintf(res_file, "%d\n", int(detections.size()));
		for (size_t i = 0; i < detections.size(); i++)
		{
		    	for(int j = 0;j < 7;j++)
			{
			    	cout<<detections[i][j]<<"    ";
			}
			cout<<endl;

			//float conf = detections[i][2];
			//float xmin = detections[i][3];
			//float ymin = detections[i][4];
			//float xmax = detections[i][5];
			//float ymax = detections[i][6];
  		        //fprintf(res_file, "%.6f    %.6f    %.6f    %.6f    %.6f\n",
			//	xmin*width, ymin*height,
			//	(xmax - xmin)*width,
			//	(ymax - ymin)*height,
			//	conf);

		        //if(conf > 0.5)
			//{
			//fprintf(res_file, "%s    %.6f    %.6f    %.6f    %.6f   \n",
			//        Img_path.c_str(), xmin*width, ymin*height,
			//	(xmax - xmin)*width,
			//	(ymax - ymin)*height
			//	);
			//}
			
			//printf("%s    %.6f    %.6f    %.6f    %.6f   \n",
			  //      Img_path.c_str(), xmin*width, ymin*height,
			//	(xmax - xmin)*width,
			//	(ymax - ymin)*height
			//	);
		}
		count++;
	}
        clock_t end_time = clock();
	double mean_time = double((end_time-start_time))/CLOCKS_PER_SEC/(count-1)*1000;
	printf("mean run time is: %f ms\n",mean_time);
	fclose(res_file);
     

}

