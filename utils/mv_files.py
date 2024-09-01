import os
import shutil

folderA = '/home/qiaoyingxv/code/ear_edge_detection/yolov5-pytorch-main/ear_datasets/xml'  # 文件夹A的路径
folderB = '/home/qiaoyingxv/code/ear_edge_detection/yolov5-pytorch-main/ear_datasets/images_jpg'  # 文件夹B的路径
folderC = '/home/qiaoyingxv/code/ear_edge_detection/yolov5-pytorch-main/ear_datasets/JPEGImages'  # 文件夹C的路径

filesA = os.listdir(folderA)  # 获取文件夹A中的所有文件名

for filenameA in filesA:
    file_base_name=filenameA[:-4]
    file_B_path=os.path.join(folderB, file_base_name +".png")
    file_C_path = os.path.join(folderC, file_base_name+'.png')
    if os.path.exists(os.path.join(folderB, file_base_name +".png")):
        shutil.copy2(file_B_path,file_C_path )
