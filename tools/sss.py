import numpy as np
import sys
sys.path.insert(0, '/data/home/larainelu/SFD/caffe/python')
import caffe
import pylab as plt
#import matplotlib.pyplot as plt

import time

caffe.set_device(2)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()





def computeData(image, deploy='deploy.prototxt'):

    net = caffe.Net('/data/home/larainelu/SFD/caffe/models/VGGNet/WIDER_FACE/SFD_trained/' + deploy,
                    '/data/home/larainelu/SFD/caffe/models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel',
                    caffe.TEST)
    #net = caffe.Net('/data/home/larainelu/ssd/caffe/models/VGGNet/VOC0712/SFD_RES26/' + deploy,
    #                '/data/home/larainelu/ssd/caffe/models/VGGNet/VOC0712/SFD_RES26/SFD_RES26.caffemodel',
    #                caffe.TEST)

    net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
    print image.shape[0], image.shape[1]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    start = time.time()
    for i in range(10):
      net.forward()
    end = time.time()
    t = (end - start) / 10
    return t


image = caffe.io.load_image('/data/home/larainelu/face.jpg')#0_Parade_marchingband_1_20


t1 = computeData(image, 'deploy.prototxt')


print t1
