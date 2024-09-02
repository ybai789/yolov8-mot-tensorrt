import os

weights = { '0': os.path.join('weights','yolov5x6.pt'),
            '1': os.path.join('weights','yolov5x6.onnx'),
            '2': os.path.join('weights','yolov5s.pt'),
            '3': os.path.join('weights','yolov5s.onnx'),
            '4': os.path.join('weights','yolov5n.pt'),
            '5': os.path.join('weights','yolov5n.onnx'),
            '6': os.path.join('weights','yolov5m.pt'),
            '7': os.path.join('weights','yolov5m.onnx'),
            '8': os.path.join('weights','yolov5l.pt'),
            '9': os.path.join('weights','yolov5l.onnx'),
            '10': os.path.join('weights','yolov5x.pt'),
            '11': os.path.join('weights','yolov5x.onnx'),
            '12': os.path.join('weights','yolov5n6.pt'),
            '13': os.path.join('weights','yolov5n6.onnx'),
            '14': os.path.join('weights','yolov5s6.pt'),
            '15': os.path.join('weights','yolov5s6.onnx'),
            '16': os.path.join('weights','yolov5m6.pt'),
            '17': os.path.join('weights','yolov5m6.onnx'),
            '18': os.path.join('weights','yolov5l6.pt'),
            '19': os.path.join('weights','yolov5l6.onnx'),

            # YOLOv8
            '20': os.path.join('weights','yolov8n.pt'),
            '21': os.path.join('weights','yolov8n.engine'),
            '22': os.path.join('weights','yolov8s.pt'),
            '23': os.path.join('weights','yolov8s.engine'),
            '24': os.path.join('weights','yolov8m.pt'),
            '25': os.path.join('weights','yolov8m.engine'),
            '26': os.path.join('weights','yolov8l.pt'),
            '27': os.path.join('weights','yolov8l.engine'),
            '28': os.path.join('weights','yolov8x.pt'),
            '29': os.path.join('weights','yolov8x.engine')

}

def get_weight_path(model_flag):
    if model_flag in range(0, 20):
        onnx = False if (model_flag % 2 == 0) else True
        weight = weights[str(model_flag)]
    elif model_flag in range(20, 30):
        onnx = False if (model_flag % 2 == 0) else True
        weight = weights[str(model_flag)]
    
    return onnx, weight
        
