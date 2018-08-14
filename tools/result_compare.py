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
    if(len(boxes1) != len(boxes2)):
        return False
    
    for i in range(len(boxes1)):
        if not check_similar_box(boxes1[i],boxes2,threshold):
            return False
    
    for i in range(len(boxes2)):
        if not check_similar_box(boxes2[i],boxes1,threshold):
            return False
        
    return True


def get_box_dict(file_name):
    file = open(file_name)
    temp = file.readline()
    m_dict = {}
    while(temp):
        box_info = temp.split()
        img_name = box_info[0]
        all_boxes = []
        position = [int(box_info[1]),int(box_info[2]),int(box_info[3]),int(box_info[4])]
        all_boxes.append(position)
        
        if(img_name in m_dict.keys()):
            m_dict[img_name].append(position)
        else:
            m_dict[img_name] = all_boxes
            
        temp = file.readline() 
    file.close()   
    return m_dict



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

    
def visualize(img_path, boxes1 , boxes2 ,outpath ):
    im = cv2.imread(img_path)
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(im, aspect='equal')
	
    for i in range(len(boxes1)):  
        ax.add_patch(
            plt.Rectangle((boxes1[i][0], boxes1[i][1]),boxes1[i][2],boxes1[i][3],fill=False,edgecolor=(0, 1, 0), linewidth=3))
	   

    for i in range(len(boxes2)):  
        ax.add_patch(
            plt.Rectangle((boxes2[i][0], boxes2[i][1]),boxes2[i][2],boxes2[i][3],fill=False,edgecolor=(0, 0 ,1), linewidth=3))

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    
    plt.savefig(os.path.join(outpath,img_path[-10:]))

    plt.clf()
    plt.cla()
    plt.close() 

def visualize_one(img_path, boxes1 , outpath ,flag ):
    im = cv2.imread(img_path)
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(im, aspect='equal')

    if flag == 1:
	for i in range(len(boxes1)):  
	    ax.add_patch(
		plt.Rectangle((boxes1[i][0], boxes1[i][1]),boxes1[i][2],boxes1[i][3],fill=False,edgecolor=(0, 1, 0), linewidth=3))
	   
    if flag == 2:
	for i in range(len(boxes1)):  
	    ax.add_patch(
		plt.Rectangle((boxes1[i][0], boxes1[i][1]),boxes1[i][2],boxes1[i][3],fill=False,edgecolor=(0, 0, 1), linewidth=3))
	   

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    
    plt.savefig(os.path.join(outpath,img_path[-10:]))
    plt.clf()
    plt.cla()
    plt.close() 



if  __name__ == '__main__':
    file1 = "./taxi.txt"
    file2 = "./taxi_my.txt"
    file3 = "./compare.txt"
    path = "./res"

    w_file = open(file3,'w')
    w_file.close()

    boxes_1 = get_box_dict(file1)
    boxes_2 = get_box_dict(file2)
    threshold = 0.5
    
    for img in boxes_1.keys():
        if img in boxes_2.keys():
            flag = compare_box(boxes_1[img], boxes_2[img], threshold)
            if flag == False:
                output_diff(img,boxes_1[img], boxes_2[img], file3)
		visualize(img,boxes_1[img],boxes_2[img],path)
        else:
            output_onlyone(img,boxes_1[img],file3,1)
	    #print("only orinal")
	    visualize_one(img,boxes_1[img],path,1)
     
    for img in boxes_2.keys():
	if not img in boxes_1.keys():
	    output_onlyone(img,boxes_2[img],file3,2)
    	    #print("only my")
	    visualize_one(img,boxes_2[img],path,2)
    
    
    

    

