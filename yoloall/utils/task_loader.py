from yoloall.detectors import YOLOv5Detector
from yoloall.detectors import YOLOv8Detector
from yoloall.trackers import ByteTrack
from yoloall.trackers import DeepSort

detectors = {
    'yolov5': YOLOv5Detector,
    'yolov8': YOLOv8Detector
}

trackers = {
    'byte_track': ByteTrack,
    'deepsort': DeepSort
}


def get_detector(detector, use_cuda=True, use_onnx=False):
    detector = detectors.get(detector, None)

    if detector is not None:
        return detector(use_cuda=use_cuda, use_onnx=use_onnx)
    else:
        return None


def get_tracker(tracker, detector, use_cuda=True, use_onnx=False):
    tracker = trackers.get(tracker, None)

    if tracker is not None:
        return tracker(detector)
    else:
        return None
