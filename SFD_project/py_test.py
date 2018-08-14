import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#def visusalize_detections(im, bboxes, plt_name='output', ext='.png', visualization_folder=None, thresh=0.5):
def visusalize_detections(img_path, xmin, ymin, xmax, ymax, score, plt_name='output', ext='.png', visualization_folder=None,  thresh=0.5):
    """
    A function to visualize the detections
    :param im: The image
    :param bboxes: The bounding box detections
    :param plt_name: The name of the plot
    :param ext: The save extension (if visualization_folder is not None)
    :param visualization_folder: The folder to save the results
    :param thresh: The detections with a score less than thresh are not visualized
    """
        
    im = cv2.imread(img_path)
    xmin = np.array(xmin)
    ymin = np.array(ymin)
    xmax = np.array(xmax)
    ymax = np.array(ymax)
    score = np.array(score)
    
    inds = np.where(score[:] >= thresh)[0]
    #print('inds:',inds)
    xmin = xmin[inds]
    xmax = xmax[inds]
    ymin = ymin[inds]
    ymax = ymax[inds]
    score = score[inds]
    
    #print(xmin)
    #print(ymin)
    #print(xmax)
    #print(ymax)
    fig, ax = plt.subplots(figsize=(12, 12))


    if im.shape[0] == 3:
        im_cp = im.copy()
        im_cp = im_cp.transpose((1, 2, 0))
        #if im.min() < 0:
        #    pixel_means = cfg.PIXEL_MEANS
        #    im_cp = im_cp + pixel_means

        im = im_cp.astype(dtype=np.uint8)

    im = im[:, :, (2, 1, 0)]

    ax.imshow(im, aspect='equal')
	
    if score.shape[0] == 0: 
        return 
		
    for i in range(score.shape[0]):  
        ax.add_patch(
            plt.Rectangle((xmin[i], ymin[i]),
                          xmax[i]-xmin[i],
                          ymax[i]-ymin[i], fill=False,
                          edgecolor=(0, score[i], 0), linewidth=3)
        )
	   

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if visualization_folder is not None:
        if not os.path.exists(visualization_folder):
            os.makedirs(visualization_folder)
        plt_name += ext
        plt.savefig(os.path.join(visualization_folder,plt_name),bbox_inches='tight')
        print('Saved {}'.format(os.path.join(visualization_folder, plt_name)))
    else:
        print('Visualizing {}!'.format(plt_name))
        plt.show()
    plt.clf()
    plt.cla()



# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {sfd_root}/sfd_test_code
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
import caffe


caffe.set_device(2)
caffe.set_mode_gpu()
#model_def = '/data/home/larainelu/SFD/caffe/models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt'
#model_weights = '/data/home/larainelu/SFD/caffe/models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel'

model_def = '/data/home/larainelu/SFD/caffe/models/VGGNet/WIDER_FACE/SFD/deploy.prototxt'
model_weights = '/data/home/larainelu/ssd/caffe/models/VGGNet/VOC0712/SFD/VGG_SFD_iter_100000.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)


count = 0
data_path = '/data/home/larainelu/DataSet/TAXI_blur/Data_test/'
f = open('/data/home/larainelu/DataSet/TAXI_blur/after_train_bbx.txt', 'wt')
for Name in open('/data/home/larainelu/DataSet/TAXI_blur/Data_test/image_list.txt'):
    Image_Path = data_path + Name[:-1] 
    image = caffe.io.load_image(Image_Path)
    heigh = image.shape[0]
    width = image.shape[1]

    net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    detections = net.forward()['detection_out']
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin =	 detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    keep_index = np.where(det_conf >= 0)[0]
    det_conf = det_conf[keep_index]
    det_xmin = det_xmin[keep_index]
    det_ymin = det_ymin[keep_index]
    det_xmax = det_xmax[keep_index]
    det_ymax = det_ymax[keep_index]

    f.write('{:s}\n'.format(Name[:-1]))
    f.write('{:.1f}\n'.format(det_conf.shape[0]))
    for i in xrange(det_conf.shape[0]):
        xmin = det_xmin[i] * width
        ymin = det_ymin[i] * heigh
        xmax = det_xmax[i] * width
        ymax = det_ymax[i] * heigh
        score = det_conf[i]
        if score > 0:
            f.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.
                    format(xmin, ymin, (xmax-xmin+1), (ymax-ymin+1), score))	


    
    #############visualize#################
    #output_path = './sfd_test_code/output/'
    #visusalize_detections(Image_Path, det_xmin * width,det_ymin * heigh,det_xmax * width,det_ymax * heigh,det_conf, plt_name = Name[:-1]+'_detection_SFD', ext='.png', visualization_folder = output_path)
    #######################################
    count += 1
    print('%d' % count)
