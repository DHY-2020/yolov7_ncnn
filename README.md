# yolov7_ncnn

1.安装opencv https://opencv.org/releases/  

2.安装ncnn   https://github.com/Tencent/ncnn  

3.导出onnx模型时需要修改export.py和yolo.py的两处，如下：  
export.py中使用onnx-simplifier简化onnx模型  
![image](https://user-images.githubusercontent.com/68861091/179701738-919a3a14-304a-45fd-9c01-dff3cee1cb39.png)  

yolo.py中修改输出方式  

![image](https://user-images.githubusercontent.com/68861091/179702312-fda93f87-c8ef-4e17-b9be-da5a60a25b45.png)  

4.下载本项目，进入项目文件夹下  

5.打开终端执行  
 mkdir build  
 cd build  
 cmake ..  
 make  
 
