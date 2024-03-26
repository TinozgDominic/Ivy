from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator 
import torch
import random

class OD():
    def __init__(self, model = "s"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained model 
        self.model_type = "m" if model == "m" else "s"
        self.model_path = f"model/object_detection/yolov8{self.model_type}.pt"
        self.model = YOLO(self.model_path)
        self.model = self.model.to(self.device)

    def detect(self, frame, iou = 0.7, conf = 0.8):
        """_summary_
        """
        results = self.model(frame, iou = iou, conf = conf, verbose = False)
        
        img = results[0].plot()
        for r in results:
            predict = [r.boxes.cls, r.boxes.conf, r.boxes.xyxy]
        
        return predict, img
    
if __name__ == "__main__":
    import numpy as np
    import cv2
    frame = np.zeros((640, 640, 3), dtype = np.uint8)
    od = OD()
    predict, img = od.detect(frame)
    cv2.imwrite("out.jpg", img)