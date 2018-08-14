import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calcIOU(x1,y1,w1,h1,x2,y2,w2,h2):
    if(x1 > x2+w2):
        return 0.
    if(y1 > y2+h2):
        return 0.
    if(x2 > x1+w1):
        return 0.
    if(y2 > y1+h1):
        return 0.

    colInt = min(x1+w1,x2+w2) - max(x1,x2)
    rowInt = min(y1+h1,y2+h2) - max(y1,y2)

    intersection = colInt*rowInt

    area1 = w1*h1
    area2 = w2*h2

    Iou = float(intersection)/(area1+area2-intersection)
    return Iou


def check_similar_box(box,boxes,threshold):
    for i in range(len(boxes)):
        temp_Iou = calcIOU(box[0],box[1],box[2],box[3],boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3])
        if(temp_Iou > threshold):
            return True
    return False


def compare_box(boxes1,boxes2, threshold):
    tp = 0

    for i in range(len(boxes1)):
        if check_similar_box(boxes1[i],boxes2,threshold):
            tp += 1
    
    return tp



def get_box_dict(file_name):
    file = open(file_name)
    m_dict = {}
    all_box_num = 0

    name = file.readline()
    while(name):
	box_num = int(float(file.readline()))
	#print(box_num)

        #all_box_num += box_num
	all_boxes = []
	for i in range(box_num):
	    position = file.readline().split()
	    position = map(int,position)
	    #if len(position) == 5:
	    #	if position[4] > threshold:
	    #	    del position[4]
	    #	else:
	    #	    continue;
	    all_boxes.append(position)
	all_box_num += len(all_boxes)
	m_dict[name] = all_boxes
	name = file.readline()

    file.close()
    return m_dict,all_box_num






def output_diff(img_name, boxes_1, boxes_2, file):
    w_file = open(file,'a')
    
    for i in range(len(boxes_1)):
        w_file.write(img_name)
        w_file.write('   ')
        temp = '    '.join('%s' %id for id in boxes_1[i])
        w_file.write(temp)
        w_file.write('   1\n')

    for i in range(len(boxes_2)):
        w_file.write(img_name)
        w_file.write('   ')
        temp = '    '.join('%s' %id for id in boxes_2[i])
        w_file.write(temp)
        w_file.write('   2\n')

    
    w_file.close()


def output_onlyone(img_name, boxes, file, flag):
    w_file = open(file,'a')
    
    for i in range(len(boxes)):
        w_file.write(img_name)
        w_file.write('   ')
        temp = '    '.join('%s' %id for id in boxes[i])
        w_file.write(temp)
        w_file.write('   %d\n'%flag)

    w_file.close()

    


if  __name__ == '__main__':
    gt_res = "/data/home/larainelu/DataSet/widerface/wider_face_split/wider_face_val_bbx_gt.txt"
    #test_res = "/data/home/larainelu/DataSet/widerface/SFD_val_detect.txt"
    test_res = "/data/home/larainelu/DataSet/widerface/SFD_RES26_val_detect.txt"


    (test_box, test_box_num) = get_box_dict(test_res)
    (gt_box, gt_box_num) = get_box_dict(gt_res)
    tp = 0
    threshold = 0.3

    index = 1
    
    for img in test_box.keys():
	print('the %dth image...' %index)
	index += 1
        if img in gt_box.keys():
            tp += compare_box(test_box[img], gt_box[img], threshold)
	    
    
    
    print('tp = ',tp)
    print('test_box_num = ',test_box_num)
    print('gt_box_num = ',gt_box_num)
 
    print('recall = ',float(tp)/gt_box_num)
    print('precision = ',float(tp)/test_box_num) 
