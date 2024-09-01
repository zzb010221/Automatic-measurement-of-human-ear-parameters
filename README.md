# Automatic measurement of human ear parameters based on object detection and landmark extraction
 
## Data preparation

Put the five collected images in the test_img folder. The default naming format is: 1_front.bmp, 1_l_left.bmp, 1_left.bmp, 1_r_right.bmp, 1_right.bmp.

<div align="center">
	<img src="/test_img/1_front.bmp" alt="Editor" width="150" >
	<img src="/test_img/1_l_left.bmp" alt="Editor" width="150">
 <img src="/test_img/1_left.bmp" alt="Editor" width="150">
 <img src="/test_img/1_r_right.bmp" alt="Editor" width="150">
 <img src="/test_img/1_right.bmp" alt="Editor" width="150">
</div>

## Operation process

Modify the serial number of the image read in detection_model_image5.py and run the program. The print box will output a series of parameters such as head height, head width and human ear, and pop up the effect pictures of iris detection, human ear detection and human ear landmark extraction.

<div align="center">
	<img src="/result/1.jpg" alt="Editor" width="150" >
	<img src="/result/2.jpg" alt="Editor" width="150" >
	<img src="/result/3.jpg" alt="Editor" width="150" >
	<img src="/result/4.jpg" alt="Editor" width="150" >
	<img src="/result/5.jpg" alt="Editor" width="150" >
</div>
