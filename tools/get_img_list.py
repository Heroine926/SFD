
# -*- coding: utf-8 -*-

"""

@author: larainelu


"""




import os
data_root = '/data/home/larainelu/DataSet/widerface/WIDER_val/images'  
os.chdir(data_root)



def writeTXT(root, f):
    g = os.walk(root)
    #f = open(filename,'wt')
    for path,d,filelist in g:  
        for filename in filelist:
            if filename.endswith('jpg'):   
                temp = os.path.join(path, filename)
                file = temp[2:] 
                templist = file.split('\\')
                con = '/'
                file = con.join(templist) 
                f.write(file)
                f.write('\n')
                print(file)
#        for subdir in d:
#            writeTXT(subdir,f)



if __name__=='__main__':
    root = './';
    f = open('./image_list.txt','wt')
    writeTXT(root,f)
    

