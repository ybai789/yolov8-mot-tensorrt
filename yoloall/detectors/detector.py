import cv2
from yoloall.detectors.yolov5 import YOLOv5Detector
from yoloall.detectors.yolov8 import YOLOv8Detector
from yoloall.detectors.utils.weights_path import get_weight_path
from .yolov8 import YOLOv8Detector


class Detector:
    def __init__(self,
                 model_flag: int,
                 weights: str = None,
                 use_cuda: bool = True,
                 num_classes=80):
        
        self.model = self._select_detector(model_flag, weights, use_cuda, num_classes)
        
    def _select_detector(self, model_flag, weights, cuda, num_classes):
        # Get required weight using model_flag

        if weights and weights.split('.')[-1] == 'engine':
            onnx = True
            weight = weights
        elif weights:
            onnx = False
            weight = weights
        else:
            onnx, weight = get_weight_path(model_flag)

        if model_flag in range(0, 20):
            _detector = YOLOv5Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        
        if model_flag in range(20, 30):
            # Get exp file and corresponding model for pytorch only
            _detector = YOLOv8Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
            
        return _detector

    def get_detector(self):
        return self.model

    def detect(self,
               image: list,
               return_image=False,
               **kwargs: dict):
        return self.model.detect(image,return_image,**kwargs)


if __name__ == '__main__':

    # Initialize YOLOv8 object detector
    model_type = 22
    result = Detector(model_flag=model_type, use_cuda=True)
    img = cv2.imread('../yoloall/data/images/bus.jpeg')
    pred = result.get_detector(img)
    print(pred)
