CAFFE := ../caffe
CAFFE_LIB_PATH := $(CAFFE)/build/lib

CAFFE_INC_DIR := $(CAFFE)/include
CAFFE_SRC_DIR := $(CAFFE)/src
CAFFE_SO_DIR := $(CAFFE)/.buDetect.so 
CAFFE_BIU_DIR := $(CAFFE)/.build_release/src
USR_INC_DIR := /usr/local/include
CUDA_INC_DIR := /usr/local/cuda/include


SO_INC_DIR = -I$(CAFFE_INC_DIR)\
	     -I$(CAFFE_SRC_DIR)\
	     -I$(CAFFE_SO_DIR)\
	     -I$(CAFFE_BIU_DIR)\
	     -I$(USR_INC_DIR)\
	     -I$(CUDA_INC_DIR)


CC = g++
SO_PATH = sfd_FaceDetect.so
H_PATH = ./soSFD.h
CPP_PATH = ./soSFD.cpp


#$(SO_PATH):$(CPP_PATH)
so:
	$(CC) -fPIC -shared $(CPP_PATH) -o $(SO_PATH) $(SO_INC_DIR) -L$(CAFFE_LIB_PATH) -lcaffe -Wl,-rpath=$(CAFFE_LIB_PATH)


PROJECT_NAME := UseSFD
MAIN_CPP := ./test_sfd.cpp

DL := dl
BOOST_SYS := boost_system
BOOST_FILESYS := boost_filesystem
OPENCV_CORE := opencv_core
OPENCV_HIGHGUI := opencv_highgui

LIB_NAME = -l$(DL)\
	   -l$(BOOST_SYS)\
	   -l$(BOOST_FILESYS)\
	   -l$(OPENCV_CORE)\
	   -l$(OPENCV_HIGHGUI)

test:
	$(CC) -g -rdynamic -o $(PROJECT_NAME) $(MAIN_CPP) $(LIB_NAME)


            
clean:
	rm -rf $(SO_PATH)


