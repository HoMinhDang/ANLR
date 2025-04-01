import math
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath('yolov5'))
from yolov5.utils.general import non_max_suppression
from yolov5.utils.dataloaders import letterbox  
from yolov5.models.experimental import attempt_load
import cv2

class CharDetector:
    def __init__(self, weights_path='char_custom.pt', device='cpu'):
        """
        Load custom model based on YOLOv5
        """
        self.device = device
        self.model = attempt_load(weights_path, device=device)
        self.model.to(device)

        label_path = './char_name.txt'
        self.names = [name.strip() for name in open(label_path).readlines()]
        self.size = 128

    def process_img(self, img):
        """
        Resize and preprocess image. Return resized image and tensor of image.
        """
        if img is None:
            raise ValueError("Image could not be read")
        
        img_resized = letterbox(img, self.size, auto=True)[0]
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img_resized = np.ascontiguousarray(img_resized) / 255.0  # Normalize 0-1
        img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0)  # Add batch dimension

        return img_resized, img_tensor

    def inference(self, img, conf_thres, iou_thres):
        """
        Run inference on image tensor. Return detections.
        """
        _, img_tensor = self.process_img(img)
        img_tensor.to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
        
        results = []
        if pred is not None and len(pred):
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = map(int, xyxy)
                xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                results.append((xc, yc, w, h, self.names[int(cls)], float(conf)))
        
        return results

    def filter_nearby_bbox(self, pred, min_distance=10):
        def calc_dist(x1, y1, x2, y2):
            return math.sqrt((x1-x2)**2 + (y1-y2)**2)

        keep = [True] * len(pred)
        for i in range(len(pred)):
            if not keep[i]:
                continue
            x1, y1, _, _, _, conf1 = pred[i]
            for j in range(len(pred)):
                if (i != j) and keep[j]:
                    x2, y2, _, _, _, conf2 = pred[j]
                    dist = calc_dist(x1, y1, x2, y2)
                    if dist < min_distance:
                        if conf1 < conf2:
                            keep[i] = False
                            break
                        else:
                            keep[j] = False
        results = [det for det, k in zip(pred, keep) if k]
        return results

    def extract_1line(self, pred):
        """
        Extract 1 line of characters from predictions.
        """
        if not pred:
            return ""
        pred = sorted(pred, key=lambda x: x[0])  # Sort by x coordinate
        return "".join(det[4] for det in pred)

    def extract_2line(self, pred):
        """
        Extract 2 lines of characters from predictions.
        """
        if not pred:
            return ""
        
        X = np.array([x[0] for x in pred])
        Y = np.array([x[1] for x in pred])
        m, b = np.linalg.lstsq(np.vstack([X, np.ones(len(X))]).T, Y, rcond=None)[0]
        
        line1, line2 = [], []
        for det in pred:
            x, y, _, _, _, _ = det
            (line1 if y < m * x + b else line2).append(det)
        
        line1.sort(key=lambda x: x[0])
        line2.sort(key=lambda x: x[0])
        return "".join(det[4] for det in line1) + " " + "".join(det[4] for det in line2)
    

    def draw_bbox(self, img, pred):
        # Sao chép ảnh đầu vào
        output_img = img.copy()
        
        # Debug ảnh đầu vào
        print("Input img - Type:", type(output_img), "Shape:", output_img.shape, "Dtype:", output_img.dtype)
        
        # Áp dụng letterbox
        letterboxed = letterbox(output_img, 128, auto=True)
        output_img = letterboxed[0]  # Lấy ảnh từ tuple (img, ratio, pad)
        
        # Debug sau letterbox
        print("After letterbox - Type:", type(output_img), "Shape:", output_img.shape, "Dtype:", output_img.dtype)
        
        # Đảm bảo kiểu dữ liệu là uint8
        if output_img.dtype != np.uint8:
            print("Converting to uint8...")
            output_img = output_img.astype(np.uint8)
        
        # Đảm bảo ảnh là 3 kênh (BGR)
        if len(output_img.shape) == 2:  # Nếu là ảnh xám
            output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
        elif output_img.shape[2] != 3:  # Nếu không phải 3 kênh
            raise ValueError("Image must have 3 channels (BGR)")
        
        # Debug cuối cùng trước khi vẽ
        print("Final img - Type:", type(output_img), "Shape:", output_img.shape, "Dtype:", output_img.dtype)
        
        # Nếu pred là [detections], lấy detections
        detections = pred[0] if isinstance(pred, list) and pred and isinstance(pred[0], list) else pred
        
        for x, y, w, h, label, conf in detections:
            # Chuyển từ tọa độ trung tâm sang góc trên trái và dưới phải
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            # Vẽ hình chữ nhật
            cv2.rectangle(output_img, 
                         (x1, y1), 
                         (x2, y2), 
                         (0, 255, 0),  # Màu xanh lá
                         2)            # Độ dày
            
            # Chuẩn bị text hiển thị label
            text = f"{label}"
            
            # Tính kích thước text để vẽ nền
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            # Vẽ nền cho text
            text_bg_x2 = x1 + text_size[0]
            text_bg_y1 = y1 - text_size[1] - 4
            cv2.rectangle(output_img, 
                         (x1, text_bg_y1), 
                         (text_bg_x2, y1), 
                         (0, 255, 0), 
                         -1)  # Filled
            
            # Vẽ text
            cv2.putText(output_img, 
                       text, 
                       (x1, y1 - 4), 
                       font, 
                       font_scale, 
                       (0, 0, 0),  # Màu đen cho text
                       font_thickness)
        
        return output_img


    def detect_char(self, img, labels, conf_thres=0.05, iou_thres=0.04, min_dist =10):
        """
        Detect characters in the image. Return recognized string.
        """
        pred = self.inference(img, conf_thres, iou_thres)
        pred = self.filter_nearby_bbox(pred, min_dist)
        ##DEBUG
        # output = self.draw_bbox(img, pred)
        # cv2.imshow("test", output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        ##DEBUG
        if not pred:
            raise ValueError("No characters detected")
        
        if labels == '1_line':
            return self.extract_1line(pred)
        elif labels == '2_line':
            return self.extract_2line(pred)
        else:
            raise ValueError(f"Type {labels} not supported")
