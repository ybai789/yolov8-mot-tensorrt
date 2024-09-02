import os
from yoloall import utils
from yoloall.utils import get_names
from yoloall.utils import tlwh_to_xyxy

import torch
from .utils.yolov8_utils import prepare_input, process_output
import numpy as np
import warnings
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import attempt_load_one_weight
import sys
import platform
from ctypes import *
import numpy.ctypeslib as npct

class YOLOv8Detector:
    def __init__(self,
                 weights=None,
                 use_onnx=False,
                 use_cuda=True):

        self.use_onnx = use_onnx
        self.device = 'cuda' if use_cuda else 'cpu'

        # If incase weighst is a list of paths then select path at first index
        weights = str(weights[0] if isinstance(weights, list) else weights)
        print(f"weights = {weights}")

        if not os.path.exists(weights):
            print(f"{weights} not exist!")
            sys.exit()   

        if self.use_onnx:
            if platform.system().lower() == 'windows':
                dll_path =  "./weights/yolov8.dll"
                self.model = CDLL(dll_path, winmode=0)
            else:
                dll_path =  "./weights/libyolov8.so"
                self.model = cdll.LoadLibrary(dll_path)
            model_path = weights
            model_path = model_path.encode()  # Convert model_path to byte string
            print(f"model_path = {model_path} dll_path = {dll_path}")
             
            self.model.Detect.argtypes = [c_void_p, c_int, c_int, POINTER(c_ubyte), npct.ndpointer(dtype=np.float32, ndim=2, shape=(100, 6), flags="C_CONTIGUOUS")]
            self.model.Init.restype = c_void_p
            self.model.Init.argtypes = [c_void_p]
            self.c_point = self.model.Init(model_path)
        else:
            # Load Model
            self.model = self.load_model(use_cuda, weights)

    def load_model(self, use_cuda, weights, fp16=False):

        # Device: CUDA and if fp16=True only then half precision floating point works
        self.fp16 = fp16 & (
            (not self.use_onnx or self.use_onnx) and self.device != 'cpu')

        model, ckpt = attempt_load_one_weight(weights)
        model = AutoBackend(model, fp16=False, dnn=False).to(self.device)
        model.half() if self.fp16 else model.float()
        return model

    def detect(self, image: list,
               input_shape: tuple = (640, 640),
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               max_det: int = 1000,
               filter_classes: bool = None,
               agnostic_nms: bool = True,
               with_p6: bool = False,
               return_image=False
               ) -> list:

        # Preprocess input image and also copying original image for later use
        original_image = image.copy()
        detection = []
        image_info = {
            'width': original_image.shape[1],
            'height': original_image.shape[0],
        }

        # Perform Inference on the Image
        if self.use_onnx:
            # Run ONNX model
            rows, cols = image_info['height'], image_info['width']
            res_arr = np.zeros((100, 6), dtype=np.float32)
            self.model.Detect(self.c_point, c_int(rows), c_int(cols), image.ctypes.data_as(POINTER(c_ubyte)), res_arr)
            detection = res_arr[~(res_arr == 0).all(1)]
            # 交换最后两个维度的位置, conf  class_id
            detection[:, [4, 5]] = detection[:, [5, 4]]
            detection[:, :4] = np.round(detection[:, :4])
            for i in range(detection.shape[0]):  # 遍历所有检测
                detection[i, :4] = tlwh_to_xyxy(detection[i, :4])
        else:
            processed_image = prepare_input(
                image, input_shape, 32, False if self.use_onnx else True)
            processed_image = torch.from_numpy(processed_image).to(self.device)
            # Change image floating point precision if fp16 set to true
            processed_image = processed_image.half() if self.fp16 else processed_image.float()

            with torch.no_grad():
                prediction = self.model(processed_image, augment=False)
                
            # Postprocess prediction
            detection = process_output(prediction,
                                    original_image.shape[:2],
                                    processed_image.shape[2:],
                                    conf_thres,
                                    iou_thres,
                                    agnostic=agnostic_nms,
                                    max_det=max_det)

        #print(f"detection = {detection}")
        if filter_classes:
            class_names = get_names()

            filter_class_idx = []
            if filter_classes:
                for _class in filter_classes:
                    if _class.lower() in class_names:
                        filter_class_idx.append(
                            class_names.index(_class.lower()))
                    else:
                        warnings.warn(
                            f"class {_class} not found in model classes list.")

            detection = detection[np.in1d(
                detection[:, 5].astype(int), filter_class_idx)]

        if return_image:
            return detection, original_image
        else: 
            return detection, image_info
        
