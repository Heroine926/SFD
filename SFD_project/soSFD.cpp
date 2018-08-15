#include<fstream>
#include<iostream>
#include<stdio.h>
#include<caffe/caffe.hpp>
#include"./soSFD.h"

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
	void initial(const string& model_def, const string& trained_file,int gpu_id, const string& mean_file, const string& mean_value);

	int detector(const vector<cv::Mat>& imgs, vector<vector<cv::Rect> >& detections);
private:
	void SetMean(const string& mean_file, const string& mean_value);

        void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const vector<cv::Mat>& img,
		std::vector<cv::Mat>* input_channels);

public:
	int batchSize;
private:

	Net<float>* net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
};







void faceDetection::initial(const string& model_def, const string& trained_file ,int gpu_id, const string& mean_file, const string& mean_value)
{
   	#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
    	#else
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(gpu_id);
    	#endif

	/* Load the network. */
	//net_.reset(new Net<float>(model_def, TEST));
	net_ = new Net<float>(model_def,TEST);
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

}


int faceDetection::detector(const vector<cv::Mat>& imgs, vector<vector<cv::Rect> >& detections)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(batchSize, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(imgs, &input_channels);

	net_->Forward();

	Blob<float>* result_blob =  net_->output_blobs()[0];
        const float* result = result_blob->cpu_data();
        const int num_det = result_blob->height();

	int k = 0;
        while ( k < num_det)
	{
                if (result[0] == -1 || result[2] < 0.5 ) 
		{
                        // Skip invalid detection.
                        result += 7;
			k++;
                        continue;
                }
		
		vector<cv::Rect> oneimg_rect;
		int index = result[0];
		int xmin = (int)(result[3]*imgs[index].cols);
		int ymin = (int)(result[4]*imgs[index].rows);
		int xmax = (int)(result[5]*imgs[index].cols);
		int ymax = (int)(result[6]*imgs[index].rows);
		int rect_width = xmax-xmin;
		int rect_height = ymax-ymin;
		//cout << "Rect: "<<xmin<<"  "<<ymin<<endl;
		cv::Rect rect(xmin,ymin,rect_width,rect_height);
		detections[index].push_back(rect);
              
		result += 7;
		k++;
        }
	//if(detections.empty()) 
	//    return 1;
        return 0;
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

	for (int i = 0; i < input_layer->channels()*input_layer->num(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void faceDetection::Preprocess(const vector<cv::Mat>& img_vec,
	std::vector<cv::Mat>* input_channels) {
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

		std::vector<cv::Mat> vec_tmp(b+i*3, b+i*3+3);
		cv::split(sample_normalized, vec_tmp);

	}
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


static faceDetection m_faceDetection;



extern "C" int sfd_face_detection_init(const string& model_file, const string& trained_file,int gpu_id)
{
    	string mean_file = "";
  	string mean_value = "104,117,123";
	m_faceDetection.initial(model_file, trained_file, gpu_id, mean_file, mean_value);
	
	return 0;
}



extern "C" int sfd_face_detection(const void *p_cvmat, void* p_vec_vec_rect, int batchSize)
{

	if (p_cvmat == NULL) return -1;
	const cv::Mat *pimg = (const cv::Mat*)p_cvmat;
	std::vector<cv::Mat> imgs;
	for (int i = 0; i < batchSize; i++) imgs.push_back(pimg[i]);


	if (imgs.empty()) return -3;
	if (p_vec_vec_rect == NULL) return -5;

	m_faceDetection.batchSize = batchSize;
	std::vector<std::vector<cv::Rect> > &out = *(std::vector<std::vector<cv::Rect> > *)p_vec_vec_rect;
	out.resize(batchSize);
	//cout<<"before detector\n";
	int res = 0;


	res = m_faceDetection.detector(imgs,out);
        //cout<<"after detector\n";

	if(res != 0) return -7;


	return 0;
}





