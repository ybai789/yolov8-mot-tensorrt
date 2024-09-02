# YOLOv8-Based Multi-Object Tracking with TensorRT Acceleration

 

## 1. Introduction

This repository offers support for YOLOv8-based and TensorRT-accelerated multi-object tracking (MOT) algorithms. Various trackers, including `ByteTrack` and `DeepSORT`, can be seamlessly integrated with `YOLOv8`. The YOLOv8 detector is optimized for performance using ONNX and TensorRT.


## 2. Clone the Repo

```shell
git clone https://github.com/ybai789/yolov8-mot-tensorrt.git
```

## 3. Installation


```shell
cd yolov8-mot-tensorrt
pip install requirements.txt
```


## 4. Generate YOLOv8 ONNX and TensorRT files

Follow the steps in the repository at [https://github.com/ybai789/yolov8-ONNX-TensorRT](https://github.com/ybai789/yolov8-ONNX-TensorRT) to convert the YOLOv8 weights into an ONNX model and a TensorRT engine, and to compile the C++ program. 

Once completed, copy the generated `yolov8s.onnx`, `yolov8s.engine`, and `libyolov8.so` files to the `yolov8-mot-tensorrt/weights` directory.

## 5. Run the accerated MOT tracker

Run `main.py` to test tracker on  video file

```
python main.py data/videos/your_video_file.mp4
```


