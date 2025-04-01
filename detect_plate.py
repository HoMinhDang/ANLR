import torch 
import cv2
import numpy as np 
import os
import pandas as pd
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

class PlateDetector:
    def __init__(self, weights, device='cpu'):
        """
        Load YOLOv5 model from local directory.
        """
        self.names = ['1_line', '2_line']
        self.model = torch.hub.load('yolov5', 'custom', path=weights, source="local")

    def load_image(self, image_path):
        """
        Load an image from a given path.
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Image path {image_path} does not exist")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image {image_path} could not be read")
        
        return image

    def predict(self, image):
        """
        Perform inference on the image tensor.
        """
        result = self.model(image)
        detections = result.xyxy[0]  # Get predictions for the first image

        return detections

    def save_cropped_images(self, image, detections, image_name):
        """
        Save cropped images of detected plates.
        """
        output_dir = os.path.join("output", image_name)
        os.makedirs(output_dir, exist_ok=True)
        
        cropped_images = []
        cropped_paths = []
        labels = []
        
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_image_path = os.path.join(output_dir, f"{image_name}_{i}.jpg")
            cv2.imwrite(cropped_image_path, cropped_image)
            cropped_images.append(cropped_image)
            labels.append(self.names[int(cls)])
            cropped_paths.append(cropped_image_path)
        
        return cropped_images, labels, cropped_paths

    def detect_plate(self, image_path):
        """
        Detect and crop number plates. Save cropped file in output/<image_path>/.
        Return cropped images and their paths.
        """
        image = self.load_image(image_path)
        detections = self.predict(image)
        
        if detections is None or len(detections) == 0:
            raise ValueError(f"No plates detected in {image_path}")
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        return self.save_cropped_images(image, detections, image_name)
