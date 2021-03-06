# SFD
此工程实现caffe下的SFD人脸检测算法。包括train、detection、模型封装、模型性能优化等。

## 环境需求：
python+caffe+cuDNN 
## 安装步骤：
此模型是基于SSD的，需先下载SSD模型：
```Bash
git clone https://github.com/weiliu89/caffe.git 
# 此模型根目录记为$SSD 
cd $SSD
```
编译caffe
```Bash
git checkout ssd cp Makefile.config.example Makefile.config 
#注释掉Makefile.config文件中的 USE_CUDNN := 1 以及WITH_PYTHON_LAYER := 1两行 
make -j8 make py make test -j8 
#这里用的caffe环境和SSD是一样的
```
下载SFD预训练好的参数模型 https://pan.baidu.com/s/1nvoW1wH （百度云盘）， 放在$SSD/models路径下。
## Detection：
提供了两种使用接口：
### python接口 
待检测的图片所在的路径默认为$SSD/sfd_test_code，可以通过修改test_demo.py中第102行Path变量的值来自定义路径 待检测的图片名称列表，默认为$SSD/sfd_test_code/demo_img_list.txt ,可以修改test_demo.py中第104行 for Name in open('path')中path的值来自定义名称列表文件 test_demo.py中第103行打开的文件是用来写入图像中框的坐标信息以及框的得分。也可以自定义。 检测后的结果图像默认保存在$SSD/sfd_test_code/output/中，图片名称默认为’原始图片名字_detection_SFD.png'。可以修改test_demo.py第146行output_path的值自定义结果保存路径；可以修改147行函数visusalize_detections的plt_name参数自定义结果图像名称；可以修改147行函数visusalize_detections的ext参数修改结果图像的格式（后缀）
```Bash
cd $SSD/tools
python test_demo.py 
```
### C接口
修改 $SSD/SFD_project/test_sfd.cpp中数据集的路径。
```Bash
cd $SSD/SFD_project
make so  # 生成动态链接库 .so文件
make test  # 编译C代码，生成可执行文件
./UseSFD  # 运行可执行文件
```
## Train
## Modified Model
