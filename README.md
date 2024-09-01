 Automatic measurement of human ear parameters based on object detection and landmark extraction
 
数据准备：将采集的五张图像放在test_img文件夹下，默认命名格式为：人脸正面1_front.bmp、人脸左侧面1_l_left.bmp、人耳左正面1_left.bmp、人脸右侧面1_r_right.bmp、人耳右正面1_right.bmp。

运行过程：修改detection_model_image5.py里读取图片的序号，运行程序，打印框会输出头高、头宽以及人耳一系列参数，同时弹出虹膜检测、人耳检测、人耳关键点检测的效果图。
