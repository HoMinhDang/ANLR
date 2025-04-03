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
        Khởi tạo mô hình nhận diện ký tự sử dụng YOLOv5.
        
        Parameters:
        - weights_path (str): Đường dẫn đến file trọng số của mô hình.
        - device (str): Thiết bị để chạy mô hình ('cpu' hoặc 'cuda').
        """
        self.device = device
        self.model = attempt_load(weights_path, device=device)
        self.model.to(device)

        # Danh sách các ký tự có thể nhận diện
        self.names = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        self.size = 128  # Kích thước ảnh đầu vào của mô hình

    def process_img(self, img):
        """
        Xử lý ảnh đầu vào: resize, chuyển đổi màu, chuẩn hóa và đưa về tensor.
        
        Parameters:
        - img (numpy.ndarray): Ảnh đầu vào.
        
        Returns:
        - img_resized (numpy.ndarray): Ảnh đã resize.
        - img_tensor (torch.Tensor): Tensor ảnh đã được chuẩn hóa.
        """
        if img is None:
            raise ValueError("Image could not be read")
        
        img_resized = letterbox(img, self.size, auto=True)[0]
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # Chuyển từ BGR sang RGB
        img_resized = np.ascontiguousarray(img_resized) / 255.0  # Chuẩn hóa về [0,1]
        img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0)  # Thêm batch dimension

        return img_resized, img_tensor

    def inference(self, img, conf_thres, iou_thres):
        """
        Chạy mô hình nhận diện ký tự trên ảnh.
        
        Parameters:
        - img (numpy.ndarray): Ảnh đầu vào.
        - conf_thres (float): Ngưỡng confidence.
        - iou_thres (float): Ngưỡng IoU cho NMS.
        
        Returns:
        - results (list): Danh sách bounding box với định dạng (xc, yc, w, h, label, confidence).
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
        """
        Lọc bỏ các bounding box gần nhau để tránh trùng lặp.
        
        Parameters:
        - pred (list): Danh sách bounding box.
        - min_distance (int): Khoảng cách tối thiểu giữa hai bbox.
        
        Returns:
        - results (list): Bounding box sau khi lọc.
        """
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
        Ghép ký tự thành một dòng theo thứ tự từ trái sang phải.
        
        Parameters:
        - pred (list): Danh sách bounding box.
        
        Returns:
        - str: Chuỗi ký tự đã nhận diện.
        """
        if not pred:
            return ""
        pred = sorted(pred, key=lambda x: x[0])  # Sắp xếp theo trục x
        return "".join(det[4] for det in pred)

    def extract_2line(self, pred):
        """
        Phân loại ký tự thành hai dòng.
        
        Parameters:
        - pred (list): Danh sách bounding box.
        
        Returns:
        - str: Chuỗi ký tự gồm hai dòng.
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

    def detect_char(self, img, labels, conf_thres=0.05, iou_thres=0.04, min_dist=10):
        """
        Nhận diện ký tự trong ảnh và trả về chuỗi đã nhận diện.
        
        Parameters:
        - img (numpy.ndarray): Ảnh đầu vào.
        - labels (str): '1_line' hoặc '2_line'.
        - conf_thres (float): Ngưỡng confidence.
        - iou_thres (float): Ngưỡng IoU.
        - min_dist (int): Khoảng cách tối thiểu giữa các bbox.
        
        Returns:
        - str: Chuỗi ký tự đã nhận diện.
        """
        pred = self.inference(img, conf_thres, iou_thres)
        pred = self.filter_nearby_bbox(pred, min_dist)
        if not pred:
            raise ValueError("No characters detected")
        
        return self.extract_1line(pred) if labels == '1_line' else self.extract_2line(pred)
